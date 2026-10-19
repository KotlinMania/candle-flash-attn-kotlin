package io.github.kotlinmania.candleflashattn

import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.CudaDevice
import io.github.kotlinmania.candle.CudaStorage
import io.github.kotlinmania.candle.CustomOp3
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Dims4
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.applyOp3
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.dims1
import io.github.kotlinmania.candle.dims4
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16

/**
 * Rounds [x] up to the nearest multiple of [m].
 */
internal fun roundMultiple(x: Int, m: Int): Int {
    return ((x + m - 1) / m) * m
}

public data class FlashAttn(
    val softmaxScale: Float,
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

        val qSlice = q.asCudaSlice<T>()
        q.asCudaSlice<T>()
        v.asCudaSlice<T>()

        val qStride = qL.stride()
        val kStride = kL.stride()
        val vStride = vL.stride()
        val oStride = outL.stride()

        val qRank = qStride.size
        val kRank = kStride.size
        val vRank = vStride.size
        val oRank = oStride.size

        if (qRank != 4 || kRank != 4 || vRank != 4) {
            bail("flash-attn expects input tensors of rank 4 (q: $qRank, k: $kRank, v: $vRank)")
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

        val qDims = qL.shape().dims4()
        val bSz = qDims.d0
        val seqlenQ = qDims.d1
        val numHeads = qDims.d2
        val headSizeOg = qDims.d3

        val kDims = kL.shape().dims4()
        val seqlenK = kDims.d1
        val numHeadsK = kDims.d2

        val expectedKv = Dims4(bSz, seqlenK, numHeadsK, headSizeOg)
        if (expectedKv != kDims) {
            bail("shape mismatch q ${qL.shape()} and k ${kL.shape()}")
        }
        if (expectedKv != vL.shape().dims4()) {
            bail("shape mismatch q ${qL.shape()} and v ${vL.shape()}")
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

        val alibiSlopesTensor = alibiSlopes
        if (alibiSlopesTensor != null) {
            if (alibiSlopesTensor.dtype() != DType.F32) {
                bail("DType mismatch alibi_slopes ${alibiSlopesTensor.dtype()}, expected ${DType.F32}")
            }
            if (numHeads != alibiSlopesTensor.layout().shape().dims1()) {
                bail("shape mismatch alibi_slopes ${alibiSlopesTensor.layout().shape()}, expected ($numHeads)")
            }
        }

        var windowSizeLeftVal = windowSizeLeft?.let { if (it <= seqlenK) it else -1 } ?: -1
        var windowSizeRightVal = windowSizeRight?.let { if (it <= seqlenK) it else -1 } ?: -1

        val headSize = roundMultiple(headSizeOg, 8)
        val headSizeRounded = roundMultiple(headSize, 32)
        val seqlenQRounded = roundMultiple(seqlenQ, 128)
        val seqlenKRounded = roundMultiple(seqlenK, 128)

        val isBf16Int = if (isBf16) 1 else 0

        val isCausal = if (windowSizeLeftVal < 0 && windowSizeRightVal == 0) 1 else 0
        if (windowSizeLeftVal < 0 && windowSizeRightVal >= 0) {
            windowSizeLeftVal = seqlenK
        }
        if (windowSizeLeftVal >= 0 && windowSizeRightVal < 0) {
            windowSizeRightVal = seqlenK
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
            qStride[0].toUInt(),
            kStride[0].toUInt(),
            vStride[0].toUInt(),
            oStride[0].toUInt(),
            0u,
            qStride[qRank - 3].toUInt(),
            kStride[kRank - 3].toUInt(),
            vStride[vRank - 3].toUInt(),
            oStride[oRank - 3].toUInt(),
            qStride[qRank - 2].toUInt(),
            kStride[kRank - 2].toUInt(),
            vStride[vRank - 2].toUInt(),
            oStride[oRank - 2].toUInt(),
            bSz.toUInt(),
            numHeads.toUInt(),
            numHeadsK.toUInt(),
            headSize.toUInt(),
            headSizeRounded.toUInt(),
            softmaxScale,
            seqlenQ.toUInt(),
            seqlenK.toUInt(),
            seqlenQRounded.toUInt(),
            seqlenKRounded.toUInt(),
            (bSz * seqlenQ).toUInt(),
            isBf16Int,
            isCausal,
            0,
            windowSizeLeftVal,
            windowSizeRightVal,
            softcap ?: 0f,
            0L,
            0u,
            -1,
            0L,
            0u,
            0,
            0L,
        )

        val dst = CudaStorage.wrapCudaSlice(qSlice, dev as? CudaDevice ?: CudaDevice())
        return Pair(dst, outShape)
    }

    override fun name(): String = "flash-attn"

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
 * Flash-attention v2 layer.
 *
 * This implements scaled dot-product attention: `softmax(Q @ K^T * softmaxScale) @ V`.
 * Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
 * than q; the number of heads in k and v has to be divisible by the number of heads in q.
 *
 * Arguments:
 * - [q]: Query tensor with shape `(batch, seqLenQ, numHeadsQ, headSize)`.
 * - [k]: Key tensor with shape `(batch, seqLenKv, numHeadsKv, headSize)`.
 * - [v]: Value tensor with shape `(batch, seqLenKv, numHeadsKv, headSize)`.
 * - [softmaxScale]: Scaling factor for dot-product attention logits.
 * - [causal]: Whether to apply causal masking.
 *
 * Returns a tensor with dimensions `(batch, seqLenQ, numHeadsQ, headSize)`.
 */
public fun flashAttn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmaxScale: Float,
    causal: Boolean,
): Tensor {
    val windowSizeLeft = null
    val windowSizeRight = if (causal) 0 else null

    val op = FlashAttn(
        softmaxScale = softmaxScale,
        alibiSlopes = null,
        windowSizeLeft = windowSizeLeft,
        windowSizeRight = windowSizeRight,
        softcap = null,
    )
    return q.applyOp3(k, v, op)
}

/**
 * Flash-attention v2 layer with attention windows.
 *
 * This implements scaled dot-product attention: `softmax(Q @ K^T * softmaxScale) @ V`.
 * Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
 * than q; the number of heads in k and v has to be divisible by the number of heads in q.
 *
 * Arguments:
 * - [q]: Query tensor with shape `(batch, seqLenQ, numHeadsQ, headSize)`.
 * - [k]: Key tensor with shape `(batch, seqLenKv, numHeadsKv, headSize)`.
 * - [v]: Value tensor with shape `(batch, seqLenKv, numHeadsKv, headSize)`.
 * - [softmaxScale]: Scaling factor for dot-product attention logits.
 * - [windowSizeLeft]: Limit left attention to value tokens.
 * - [windowSizeRight]: Limit right attention to value tokens.
 *
 * Setting `windowSizeLeft = null` with `windowSizeRight = 0` applies a causal mask to the result
 * of `Q @ K^T`.
 *
 * Returns a tensor with dimensions `(batch, seqLenQ, numHeadsQ, headSize)`.
 */
public fun flashAttnWindowed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmaxScale: Float,
    windowSizeLeft: Int?,
    windowSizeRight: Int?,
): Tensor {
    val op = FlashAttn(
        softmaxScale = softmaxScale,
        alibiSlopes = null,
        windowSizeLeft = windowSizeLeft,
        windowSizeRight = windowSizeRight,
        softcap = null,
    )
    return q.applyOp3(k, v, op)
}

/**
 * Flash-attention v2 layer with ALiBi positional embeddings.
 *
 * Arguments:
 * - [q]: Query tensor with shape `(batch, seqLenQ, numHeadsQ, headSize)`.
 * - [k]: Key tensor with shape `(batch, seqLenKv, numHeadsKv, headSize)`.
 * - [v]: Value tensor with shape `(batch, seqLenKv, numHeadsKv, headSize)`.
 * - [alibiSlopes]: Alibi slopes tensor with shape `(numHeadsQ)`.
 * - [softmaxScale]: Scaling factor for dot-product attention logits.
 * - [causal]: Whether to apply causal masking.
 *
 * Returns a tensor with dimensions `(batch, seqLenQ, numHeadsQ, headSize)`.
 */
public fun flashAttnAlibi(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alibiSlopes: Tensor,
    softmaxScale: Float,
    causal: Boolean,
): Tensor {
    val windowSizeLeft = null
    val windowSizeRight = if (causal) 0 else null

    val op = FlashAttn(
        softmaxScale = softmaxScale,
        alibiSlopes = alibiSlopes.clone(),
        windowSizeLeft = windowSizeLeft,
        windowSizeRight = windowSizeRight,
        softcap = null,
    )
    return q.applyOp3(k, v, op)
}

/**
 * Flash-attention v2 layer with ALiBi positional embeddings and attention windows.
 *
 * Arguments:
 * - [q]: Query tensor with shape `(batch, seqLenQ, numHeadsQ, headSize)`.
 * - [k]: Key tensor with shape `(batch, seqLenKv, numHeadsKv, headSize)`.
 * - [v]: Value tensor with shape `(batch, seqLenKv, numHeadsKv, headSize)`.
 * - [alibiSlopes]: Alibi slopes tensor with shape `(numHeadsQ)`.
 * - [softmaxScale]: Scaling factor for dot-product attention logits.
 * - [windowSizeLeft]: Limit left attention to value tokens.
 * - [windowSizeRight]: Limit right attention to value tokens.
 *
 * Returns a tensor with dimensions `(batch, seqLenQ, numHeadsQ, headSize)`.
 */
public fun flashAttnAlibiWindowed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alibiSlopes: Tensor,
    softmaxScale: Float,
    windowSizeLeft: Int?,
    windowSizeRight: Int?,
): Tensor {
    val op = FlashAttn(
        softmaxScale = softmaxScale,
        alibiSlopes = alibiSlopes.clone(),
        windowSizeLeft = windowSizeLeft,
        windowSizeRight = windowSizeRight,
        softcap = null,
    )
    return q.applyOp3(k, v, op)
}

/**
 * Flash-attention v2 layer with optional ALiBi slopes, windowing, and logit softcapping.
 *
 * Arguments:
 * - [q]: Query tensor with shape `(batch, seqLenQ, numHeadsQ, headSize)`.
 * - [k]: Key tensor with shape `(batch, seqLenKv, numHeadsKv, headSize)`.
 * - [v]: Value tensor with shape `(batch, seqLenKv, numHeadsKv, headSize)`.
 * - [alibiSlopes]: Optional ALiBi slopes tensor with shape `(numHeadsQ)`.
 * - [softmaxScale]: Scaling factor for the softmax operation.
 * - [windowSizeLeft]: Optional limit on left attention to value tokens.
 * - [windowSizeRight]: Optional limit on right attention to value tokens.
 * - [softcap]: Softcap value for attention logits before softmax.
 *
 * Returns a tensor with dimensions `(batch, seqLenQ, numHeadsQ, headSize)`.
 */
public fun flashAttnAlibiWindowedSoftcap(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alibiSlopes: Tensor?,
    softmaxScale: Float,
    windowSizeLeft: Int?,
    windowSizeRight: Int?,
    softcap: Float,
): Tensor {
    val op = FlashAttn(
        softmaxScale = softmaxScale,
        alibiSlopes = alibiSlopes?.clone(),
        windowSizeLeft = windowSizeLeft,
        windowSizeRight = windowSizeRight,
        softcap = softcap,
    )
    return q.applyOp3(k, v, op)
}
