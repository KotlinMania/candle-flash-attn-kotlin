package io.github.kotlinmania.candleflashattn

import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.CudaDevice
import io.github.kotlinmania.candle.CudaStorage
import io.github.kotlinmania.candle.CustomOp3
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Dims3
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.applyOp3
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.dims1
import io.github.kotlinmania.candle.dims3
import io.github.kotlinmania.candle.dims4
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16

internal data class FlashAttnVarLen(
    val softmaxScale: Float,
    val maxSeqlenQ: Int,
    val maxSeqlenK: Int,
    val seqlensQ: Tensor,
    val seqlensK: Tensor,
    val blockTable: Tensor? = null,
    val mmPrefixRanges: Tensor? = null,
    val pageBlockSize: Int? = null,
    val alibiSlopes: Tensor? = null,
    val windowSizeLeft: Int? = null,
    val windowSizeRight: Int? = null,
    val softcap: Float? = null,
) : CustomOp3 {
    fun <T> cudaFwdT(
        q: CudaStorage,
        qL: Layout,
        k: CudaStorage,
        kL: Layout,
        v: CudaStorage,
        vL: Layout,
        isBf16: Boolean,
    ): Pair<CudaStorage, Shape> {
        val dev = q.device()
        val outShape = qL.shape().clone()
        val outL = Layout.contiguous(outShape)

        val seqlensQLayout = seqlensQ.layout()
        val seqlensKLayout = seqlensK.layout()

        val paged = blockTable != null

        val qSlice = q.asCudaSlice<T>()
        k.asCudaSlice<T>()
        v.asCudaSlice<T>()

        val qStride = qL.stride()
        val kStride = kL.stride()
        val vStride = vL.stride()
        val oStride = outL.stride()

        val qRank = qStride.size
        val kRank = kStride.size
        val vRank = vStride.size
        val oRank = oStride.size

        if (qRank != 3 || (!paged && kRank != 3) || (!paged && vRank != 3)) {
            bail("flash-attn-varlen expects input tensors of rank 3 (q: $qRank, k: $kRank, v: $vRank)")
        }
        if (paged && (kRank != 4 || vRank != 4)) {
            bail("flash-attn-varlen paged expects k/v tensors of rank 4 (k: $kRank, v: $vRank)")
        }
        if (qStride[qRank - 1] != 1) {
            bail("the last dim of q must be contiguous $qStride")
        }
        if (kStride[kRank - 1] != 1) {
            bail("the last dim of k must be contiguous $kStride")
        }
        if (vStride[vRank - 1] != 1) {
            bail("the last dim of v must be contiguous $vStride")
        }

        val qDims = qL.shape().dims3()
        val totalQ = qDims.d0
        val numHeads = qDims.d1
        val headSizeOg = qDims.d2

        val (numHeadsK, actualPageBlockSize) = if (paged) {
            val kDims = kL.shape().dims4()
            val pageBlockSize = kDims.d1
            val numHeadsKVal = kDims.d2
            val kHeadSize = kDims.d3
            val expectedV = kL.shape().dims4()
            if (expectedV != vL.shape().dims4()) {
                bail("shape mismatch k ${kL.shape()} and v ${vL.shape()}")
            }
            if (kHeadSize != headSizeOg) {
                bail("shape mismatch q ${qL.shape()} and k ${kL.shape()}")
            }
            val pageBlockSizeArg = this.pageBlockSize
                ?: bail("paged flash-attn requires page_block_size")
            if (pageBlockSizeArg != pageBlockSize) {
                bail("page_block_size $pageBlockSizeArg does not match k shape ${kL.shape()}")
            }
            if (pageBlockSize % 32 != 0) {
                bail("paged flash-attn requires page_block_size to be a multiple of 32 (got $pageBlockSize)")
            }
            if (headSizeOg > 512) {
                bail("paged flash-attn supports head sizes up to 512 (got $headSizeOg)")
            }
            Pair(numHeadsKVal, pageBlockSizeArg)
        } else {
            val kDims = kL.shape().dims3()
            val totalK = kDims.d0
            val numHeadsKVal = kDims.d1
            val expectedKv = Dims3(totalK, numHeadsKVal, headSizeOg)
            if (expectedKv != kL.shape().dims3()) {
                bail("shape mismatch q ${qL.shape()} and k ${kL.shape()}")
            }
            if (expectedKv != vL.shape().dims3()) {
                bail("shape mismatch q ${qL.shape()} and v ${vL.shape()}")
            }
            Pair(numHeadsKVal, 0)
        }

        if (headSizeOg > 512) {
            bail("only supports head dimension at most 512 (got $headSizeOg)")
        }
        if (headSizeOg % 8 != 0) {
            bail("only supports head sizes that are a multiple of 8 (got $headSizeOg)")
        }
        if (numHeads % numHeadsK != 0) {
            bail("number of k/v heads $numHeadsK must divide number of heads in query $numHeads")
        }

        val nseqlensQ = seqlensQLayout.shape().dims1()
        if (nseqlensQ < 2) {
            bail("seqlens_q should have a len >= 2 $nseqlensQ")
        }
        val nseqlensK = seqlensKLayout.shape().dims1()
        if (nseqlensK != nseqlensQ) {
            bail("seqlens_q and seqlens_k should have the same number of elements $nseqlensQ <> $nseqlensK")
        }

        val batchSize = nseqlensQ - 1

        val mmPrefixRangesTensor = mmPrefixRanges
        if (mmPrefixRangesTensor != null) {
            val layout = mmPrefixRangesTensor.layout()
            if (mmPrefixRangesTensor.dtype() != DType.I32) {
                bail("mm_prefix_ranges must be i32, got ${mmPrefixRangesTensor.dtype()}")
            }
            val mmDims = layout.shape().dims3()
            val mmBatch = mmDims.d0
            val two = mmDims.d2
            if (mmBatch != batchSize || two != 2) {
                bail("mm_prefix_ranges shape must be ($batchSize, max_ranges, 2), got ${layout.shape()}")
            }
            if (layout.stride().lastOrNull() != 1) {
                bail("mm_prefix_ranges last dimension must be contiguous")
            }
        }

        val alibiSlopesTensor = alibiSlopes
        if (alibiSlopesTensor != null) {
            if (alibiSlopesTensor.dtype() != DType.F32) {
                bail("DType mismatch alibi_slopes ${alibiSlopesTensor.dtype()}, expected ${DType.F32}")
            }
            if (numHeads != alibiSlopesTensor.layout().shape().dims1()) {
                bail("shape mismatch alibi_slopes ${alibiSlopesTensor.layout().shape()}, expected ($numHeads)")
            }
        }

        var windowSizeLeftVal = windowSizeLeft?.let { if (it <= maxSeqlenK) it else -1 } ?: -1
        var windowSizeRightVal = windowSizeRight?.let { if (it <= maxSeqlenK) it else -1 } ?: -1
        if (mmPrefixRangesTensor != null && windowSizeLeftVal < 0 && windowSizeRightVal == 0) {
            windowSizeLeftVal = maxSeqlenK
        }

        val headSize = roundMultiple(headSizeOg, 8)
        val headSizeRounded = roundMultiple(headSize, 32)
        val seqlenQRounded = roundMultiple(maxSeqlenQ, 128)
        val seqlenKRounded = roundMultiple(maxSeqlenK, 128)

        val isBf16Int = if (isBf16) 1 else 0

        val isCausal = if (windowSizeLeftVal < 0 && windowSizeRightVal == 0) 1 else 0
        if (windowSizeLeftVal < 0 && windowSizeRightVal >= 0) {
            windowSizeLeftVal = maxSeqlenK
        }
        if (windowSizeLeftVal >= 0 && windowSizeRightVal < 0) {
            windowSizeRightVal = maxSeqlenK
        }

        runMha(
            0L,
            0L,
            0L,
            0L,
            0L,
            0L,
            0L,
            0L,
            0u,
            if (paged) kStride[0].toUInt() else 0u,
            if (paged) vStride[0].toUInt() else 0u,
            0u,
            0u,
            qStride[qRank - 3].toUInt(),
            kStride[kRank - 3].toUInt(),
            vStride[vRank - 3].toUInt(),
            oStride[oRank - 3].toUInt(),
            qStride[qRank - 2].toUInt(),
            kStride[kRank - 2].toUInt(),
            vStride[vRank - 2].toUInt(),
            oStride[oRank - 2].toUInt(),
            batchSize.toUInt(),
            numHeads.toUInt(),
            numHeadsK.toUInt(),
            headSize.toUInt(),
            headSizeRounded.toUInt(),
            softmaxScale,
            maxSeqlenQ.toUInt(),
            maxSeqlenK.toUInt(),
            seqlenQRounded.toUInt(),
            seqlenKRounded.toUInt(),
            totalQ.toUInt(),
            isBf16Int,
            isCausal,
            1,
            windowSizeLeftVal,
            windowSizeRightVal,
            softcap ?: 0f,
            0L,
            0u,
            actualPageBlockSize,
            0L,
            0u,
            0,
            0L,
        )

        val dst = CudaStorage.wrapCudaSlice(qSlice, dev as? CudaDevice ?: CudaDevice())
        return Pair(dst, outShape)
    }

    override fun name(): String = "flash-attn-varlen"

    override fun cpuFwd(
        s1: CpuStorage,
        l1: Layout,
        s2: CpuStorage,
        l2: Layout,
        s3: CpuStorage,
        l3: Layout,
    ): Pair<CpuStorage, Shape> {
        if (s1.hashCode() == 0 || l1.hashCode() == 0 || s2.hashCode() == 0 || l2.hashCode() == 0 || s3.hashCode() == 0 || l3.hashCode() == 0) {
            bail("no cpu support for flash-attn")
        }
        bail("no cpu support for flash-attn")
    }

    override fun cudaFwd(
        s1: CudaStorage,
        l1: Layout,
        s2: CudaStorage,
        l2: Layout,
        s3: CudaStorage,
        l3: Layout,
    ): Pair<CudaStorage, Shape> {
        return when (s1.dtype()) {
            DType.F16 -> cudaFwdT<F16>(s1, l1, s2, l2, s3, l3, false)
            DType.BF16 -> cudaFwdT<Bf16>(s1, l1, s2, l2, s3, l3, true)
            else -> bail("flash-attn is only supported for f16/bf16 (${s1.dtype()})")
        }
    }
}

/**
 * Flash-attention v2 layer with variable-length batching.
 *
 * This implements scaled dot-product attention: `softmax(Q @ K^T * softmaxScale) @ V`.
 * Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
 * than q; the number of heads in k and v has to be divisible by the number of heads in q.
 *
 * Arguments:
 * - [q]: Query tensor with shape `(totalQ, numHeadsQ, headSize)`.
 * - [k]: Key tensor with shape `(totalKv, numHeadsKv, headSize)`.
 * - [v]: Value tensor with shape `(totalKv, numHeadsKv, headSize)`.
 * - [seqlensQ]: Cumulative sequence lengths in the batch for q.
 * - [seqlensK]: Cumulative sequence lengths in the batch for k and v.
 * - [maxSeqlenQ]: Maximum query sequence length for q in the batch.
 * - [maxSeqlenK]: Maximum query sequence length for k and v in the batch.
 * - [softmaxScale]: Scaling factor for dot-product attention logits.
 * - [causal]: Whether to apply causal masking.
 *
 * Returns a tensor with dimensions `(totalQ, numHeadsQ, headSize)`.
 */
public fun flashAttnVarlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seqlensQ: Tensor,
    seqlensK: Tensor,
    maxSeqlenQ: Int,
    maxSeqlenK: Int,
    softmaxScale: Float,
    causal: Boolean,
): Tensor {
    val windowSizeLeft = null
    val windowSizeRight = if (causal) 0 else null

    val op = FlashAttnVarLen(
        softmaxScale = softmaxScale,
        maxSeqlenQ = maxSeqlenQ,
        maxSeqlenK = maxSeqlenK,
        seqlensQ = seqlensQ.clone(),
        seqlensK = seqlensK.clone(),
        blockTable = null,
        mmPrefixRanges = null,
        pageBlockSize = null,
        alibiSlopes = null,
        windowSizeLeft = windowSizeLeft,
        windowSizeRight = windowSizeRight,
        softcap = null,
    )
    return q.applyOp3(k, v, op)
}

/**
 * Flash-attention v2 layer with variable-length batching and attention windows.
 *
 * Arguments:
 * - [q]: Query tensor with shape `(totalQ, numHeadsQ, headSize)`.
 * - [k]: Key tensor with shape `(totalKv, numHeadsKv, headSize)`.
 * - [v]: Value tensor with shape `(totalKv, numHeadsKv, headSize)`.
 * - [seqlensQ]: Cumulative sequence lengths in the batch for q.
 * - [seqlensK]: Cumulative sequence lengths in the batch for k and v.
 * - [maxSeqlenQ]: Maximum query sequence length for q in the batch.
 * - [maxSeqlenK]: Maximum query sequence length for k and v in the batch.
 * - [softmaxScale]: Scaling factor for dot-product attention logits.
 * - [windowSizeLeft]: Limit left attention to value tokens.
 * - [windowSizeRight]: Limit right attention to value tokens.
 *
 * Returns a tensor with dimensions `(totalQ, numHeadsQ, headSize)`.
 */
public fun flashAttnVarlenWindowed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seqlensQ: Tensor,
    seqlensK: Tensor,
    maxSeqlenQ: Int,
    maxSeqlenK: Int,
    softmaxScale: Float,
    windowSizeLeft: Int?,
    windowSizeRight: Int?,
): Tensor {
    val op = FlashAttnVarLen(
        softmaxScale = softmaxScale,
        maxSeqlenQ = maxSeqlenQ,
        maxSeqlenK = maxSeqlenK,
        seqlensQ = seqlensQ.clone(),
        seqlensK = seqlensK.clone(),
        blockTable = null,
        mmPrefixRanges = null,
        pageBlockSize = null,
        alibiSlopes = null,
        windowSizeLeft = windowSizeLeft,
        windowSizeRight = windowSizeRight,
        softcap = null,
    )
    return q.applyOp3(k, v, op)
}

/**
 * Flash-attention v2 layer with variable-length batching and paged K/V cache.
 *
 * Arguments:
 * - [q]: Query tensor with shape `(totalQ, numHeadsQ, headSize)`.
 * - [k]: Paged key tensor with shape `(numBlocks, pageBlockSize, numHeadsKv, headSize)`.
 * - [v]: Paged value tensor with shape `(numBlocks, pageBlockSize, numHeadsKv, headSize)`.
 * - [seqlensQ]: Cumulative sequence lengths in the batch for q.
 * - [seqlensK]: Cumulative sequence lengths in the batch for k and v.
 * - [blockTable]: Physical block indices with shape `(batchSize, maxBlocks)`.
 * - [mmPrefixRanges]: Optional per-batch ranges bypassing causal/window masks.
 * - [maxSeqlenQ]: Maximum query sequence length for q in the batch.
 * - [maxSeqlenK]: Maximum query sequence length for k and v in the batch.
 * - [softmaxScale]: Scaling factor for dot-product attention logits.
 * - [windowSizeLeft]: Limit left attention to value tokens.
 * - [windowSizeRight]: Limit right attention to value tokens.
 * - [pageBlockSize]: Number of tokens per K/V cache block.
 * - [softcap]: Optional logit softcapping value before softmax.
 *
 * Returns a tensor with dimensions `(totalQ, numHeadsQ, headSize)`.
 */
public fun flashAttnVarlenPagedWindowed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seqlensQ: Tensor,
    seqlensK: Tensor,
    blockTable: Tensor,
    mmPrefixRanges: Tensor?,
    maxSeqlenQ: Int,
    maxSeqlenK: Int,
    softmaxScale: Float,
    windowSizeLeft: Int?,
    windowSizeRight: Int?,
    pageBlockSize: Int,
    softcap: Float?,
): Tensor {
    val op = FlashAttnVarLen(
        softmaxScale = softmaxScale,
        maxSeqlenQ = maxSeqlenQ,
        maxSeqlenK = maxSeqlenK,
        seqlensQ = seqlensQ.clone(),
        seqlensK = seqlensK.clone(),
        blockTable = blockTable.clone(),
        mmPrefixRanges = mmPrefixRanges?.clone(),
        pageBlockSize = pageBlockSize,
        alibiSlopes = null,
        windowSizeLeft = windowSizeLeft,
        windowSizeRight = windowSizeRight,
        softcap = softcap,
    )
    return q.applyOp3(k, v, op)
}

/**
 * Flash-attention v2 layer with variable-length batching and ALiBi positional embeddings.
 *
 * Arguments:
 * - [q]: Query tensor with shape `(totalQ, numHeadsQ, headSize)`.
 * - [k]: Key tensor with shape `(totalKv, numHeadsKv, headSize)`.
 * - [v]: Value tensor with shape `(totalKv, numHeadsKv, headSize)`.
 * - [alibiSlopes]: ALiBi slopes tensor with shape `(numHeadsQ)`.
 * - [seqlensQ]: Cumulative sequence lengths in the batch for q.
 * - [seqlensK]: Cumulative sequence lengths in the batch for k and v.
 * - [maxSeqlenQ]: Maximum query sequence length for q in the batch.
 * - [maxSeqlenK]: Maximum query sequence length for k and v in the batch.
 * - [softmaxScale]: Scaling factor for dot-product attention logits.
 * - [causal]: Whether to apply causal masking.
 *
 * Returns a tensor with dimensions `(totalQ, numHeadsQ, headSize)`.
 */
public fun flashAttnVarlenAlibi(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alibiSlopes: Tensor,
    seqlensQ: Tensor,
    seqlensK: Tensor,
    maxSeqlenQ: Int,
    maxSeqlenK: Int,
    softmaxScale: Float,
    causal: Boolean,
): Tensor {
    val windowSizeLeft = null
    val windowSizeRight = if (causal) 0 else null

    val op = FlashAttnVarLen(
        softmaxScale = softmaxScale,
        maxSeqlenQ = maxSeqlenQ,
        maxSeqlenK = maxSeqlenK,
        seqlensQ = seqlensQ.clone(),
        seqlensK = seqlensK.clone(),
        blockTable = null,
        mmPrefixRanges = null,
        pageBlockSize = null,
        alibiSlopes = alibiSlopes.clone(),
        windowSizeLeft = windowSizeLeft,
        windowSizeRight = windowSizeRight,
        softcap = null,
    )
    return q.applyOp3(k, v, op)
}

/**
 * Flash-attention v2 layer with variable-length batching, ALiBi embeddings, and windows.
 *
 * Arguments:
 * - [q]: Query tensor with shape `(totalQ, numHeadsQ, headSize)`.
 * - [k]: Key tensor with shape `(totalKv, numHeadsKv, headSize)`.
 * - [v]: Value tensor with shape `(totalKv, numHeadsKv, headSize)`.
 * - [alibiSlopes]: ALiBi slopes tensor with shape `(numHeadsQ)`.
 * - [seqlensQ]: Cumulative sequence lengths in the batch for q.
 * - [seqlensK]: Cumulative sequence lengths in the batch for k and v.
 * - [maxSeqlenQ]: Maximum query sequence length for q in the batch.
 * - [maxSeqlenK]: Maximum query sequence length for k and v in the batch.
 * - [softmaxScale]: Scaling factor for dot-product attention logits.
 * - [windowSizeLeft]: Limit left attention to value tokens.
 * - [windowSizeRight]: Limit right attention to value tokens.
 *
 * Returns a tensor with dimensions `(totalQ, numHeadsQ, headSize)`.
 */
public fun flashAttnVarlenAlibiWindowed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alibiSlopes: Tensor,
    seqlensQ: Tensor,
    seqlensK: Tensor,
    maxSeqlenQ: Int,
    maxSeqlenK: Int,
    softmaxScale: Float,
    windowSizeLeft: Int?,
    windowSizeRight: Int?,
): Tensor {
    val op = FlashAttnVarLen(
        softmaxScale = softmaxScale,
        maxSeqlenQ = maxSeqlenQ,
        maxSeqlenK = maxSeqlenK,
        seqlensQ = seqlensQ.clone(),
        seqlensK = seqlensK.clone(),
        blockTable = null,
        mmPrefixRanges = null,
        pageBlockSize = null,
        alibiSlopes = alibiSlopes.clone(),
        windowSizeLeft = windowSizeLeft,
        windowSizeRight = windowSizeRight,
        softcap = null,
    )
    return q.applyOp3(k, v, op)
}

/**
 * Flash-attention v2 layer with variable-length batching, ALiBi embeddings, windows, and softcapping.
 *
 * Arguments:
 * - [q]: Query tensor with shape `(totalQ, numHeadsQ, headSize)`.
 * - [k]: Key tensor with shape `(totalKv, numHeadsKv, headSize)`.
 * - [v]: Value tensor with shape `(totalKv, numHeadsKv, headSize)`.
 * - [alibiSlopes]: Optional ALiBi slopes tensor with shape `(numHeadsQ)`.
 * - [seqlensQ]: Cumulative sequence lengths in the batch for q.
 * - [seqlensK]: Cumulative sequence lengths in the batch for k and v.
 * - [maxSeqlenQ]: Maximum query sequence length for q in the batch.
 * - [maxSeqlenK]: Maximum query sequence length for k and v in the batch.
 * - [softmaxScale]: Scaling factor for the softmax operation.
 * - [windowSizeLeft]: Optional limit on left attention to value tokens.
 * - [windowSizeRight]: Optional limit on right attention to value tokens.
 * - [softcap]: Softcap value for attention logits before softmax.
 *
 * Returns a tensor with dimensions `(totalQ, numHeadsQ, headSize)`.
 */
public fun flashAttnVarlenAlibiWindowedSoftcap(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alibiSlopes: Tensor?,
    seqlensQ: Tensor,
    seqlensK: Tensor,
    maxSeqlenQ: Int,
    maxSeqlenK: Int,
    softmaxScale: Float,
    windowSizeLeft: Int?,
    windowSizeRight: Int?,
    softcap: Float,
): Tensor {
    val op = FlashAttnVarLen(
        softmaxScale = softmaxScale,
        maxSeqlenQ = maxSeqlenQ,
        maxSeqlenK = maxSeqlenK,
        seqlensQ = seqlensQ.clone(),
        seqlensK = seqlensK.clone(),
        blockTable = null,
        mmPrefixRanges = null,
        pageBlockSize = null,
        alibiSlopes = alibiSlopes?.clone(),
        windowSizeLeft = windowSizeLeft,
        windowSizeRight = windowSizeRight,
        softcap = softcap,
    )
    return q.applyOp3(k, v, op)
}
