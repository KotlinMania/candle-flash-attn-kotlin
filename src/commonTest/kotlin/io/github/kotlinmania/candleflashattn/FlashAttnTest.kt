// port-lint: tests tests/flash_attn_tests.rs
package io.github.kotlinmania.candleflashattn

import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.dims3
import io.github.kotlinmania.candlenn.softmax
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

private fun toVec3Round(t: Tensor, digits: Int): List<List<List<Float>>> =
    io.github.kotlinmania.candle.toVec3Round(t, digits)

private fun faAcausal(q: Tensor, k: Tensor, v: Tensor, softmaxScale: Float): Tensor {
    val inDtype = q.dtype()
    val qF32 = q.toDtype(DType.F32)
    val kF32 = k.toDtype(DType.F32)
    val vF32 = v.toDtype(DType.F32)
    val att = qF32.matmul(kF32.t()).mul(softmaxScale.toDouble())
    val attSoftmax = softmax(att, -1)
    val output = attSoftmax.matmul(vF32.contiguous()).toDtype(inDtype)
    return output
}

private fun faAcausalSoftcap(q: Tensor, k: Tensor, v: Tensor, softcap: Float): Tensor {
    val inDtype = q.dtype()
    val qF32 = q.toDtype(DType.F32)
    val kF32 = k.toDtype(DType.F32)
    val vF32 = v.toDtype(DType.F32)
    val att = qF32.matmul(kF32.t())
    val scaled = (att / softcap.toDouble()).tanh() * softcap.toDouble()
    val attSoftmax = softmax(scaled, -1)
    return attSoftmax.matmul(vF32.contiguous()).toDtype(inDtype)
}

private fun faWindowedMmPrefix(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmaxScale: Float,
    window: Int,
    ranges: List<Pair<Int, Int>>,
): Tensor {
    val inDtype = q.dtype()
    val qF32 = q.toDtype(DType.F32)
    val kF32 = k.toDtype(DType.F32)
    val vF32 = v.toDtype(DType.F32)
    val qDims = q.dims3()
    val seqLen = qDims.d0
    val nHeads = qDims.d1
    val kDims = k.dims3()
    val nKvHeads = kDims.d1
    val groups = nHeads / nKvHeads
    val heads = ArrayList<Tensor>(nHeads)
    for (head in 0 until nHeads) {
        val kvHead = head / groups
        val qH = qF32.narrow(1, head, 1).squeeze(1).contiguous()
        val kH = kF32.narrow(1, kvHead, 1).squeeze(1).contiguous()
        val vH = vF32.narrow(1, kvHead, 1).squeeze(1).contiguous()
        val mask = ArrayList<Float>(seqLen * seqLen)
        for (qIdx in 0 until seqLen) {
            for (kIdx in 0 until seqLen) {
                val mmPrefix = ranges.any { (start, end) ->
                    qIdx in start until end && kIdx in start until end
                }
                val future = kIdx > qIdx
                val tooOld = qIdx >= window && kIdx <= qIdx - window
                mask.add(
                    if ((future || tooOld) && !mmPrefix) {
                        Float.NEGATIVE_INFINITY
                    } else {
                        0.0f
                    },
                )
            }
        }
        val maskTensor = Tensor.fromVec(mask, Shape(seqLen, seqLen), q.device())
        val att = qH.matmul(kH.t()).mul(softmaxScale.toDouble()) + maskTensor
        val attSoftmax = softmax(att, -1)
        heads.add(attSoftmax.matmul(vH.contiguous()))
    }
    return Tensor.stack(heads, 1).toDtype(inDtype)
}

private fun pagedShuffledBlockTable(headDim: Int) {
    val device = try {
        Device.newCuda(0)
    } catch (_: CandleException) {
        return
    }
    val seqLen = 296
    val nHeads = 8
    val nKvHeads = 2
    val blockSize = 32
    val numBlocks = (seqLen + blockSize - 1) / blockSize
    val poolBlocks = numBlocks + 7
    val perm = (0 until numBlocks).map { i -> (i * 5 + 3) % poolBlocks }
    val elemCount = seqLen * nHeads * headDim
    val qData = (0 until elemCount).map { i -> ((i % 251).toFloat() - 125.0f) / 125.0f }
    val q = try {
        Tensor.fromVec(qData, Shape(seqLen, nHeads, headDim), device).toDtype(DType.BF16)
    } catch (_: CandleException) {
        return
    }
    val kvElemCount = seqLen * nKvHeads * headDim
    val kvData = (0 until kvElemCount).map { i -> ((i % 193).toFloat() - 96.0f) / 96.0f }
    val kv = Tensor.fromVec(kvData, Shape(seqLen, nKvHeads, headDim), device).toDtype(DType.BF16)
    val kSeq = kv / 4.0
    val vSeq = kv / 5.0

    val blockElems = blockSize * nKvHeads * headDim
    fun scatter(seq: Tensor): Tensor {
        val data = seq.toDtype(DType.F32).flattenAll().toVec1<Float>()
        val pool = FloatArray(poolBlocks * blockElems)
        for (logical in perm.indices) {
            val physical = perm[logical]
            val src = logical * blockElems
            val len = minOf(blockElems, data.size - src)
            for (offset in 0 until len) {
                pool[physical * blockElems + offset] = data[src + offset]
            }
        }
        return Tensor.fromVec(pool.toList(), Shape(poolBlocks, blockSize, nKvHeads, headDim), device).toDtype(DType.BF16)
    }
    val kPaged = scatter(kSeq)
    val vPaged = scatter(vSeq)

    val seqlens = Tensor.new(uintArrayOf(0u, seqLen.toUInt()), Shape(2), device)
    val blockTable = Tensor.new(perm.map { it.toUInt() }.toUIntArray(), Shape(numBlocks), device)
        .reshape(Shape(1, numBlocks))

    val ysRef = faWindowedMmPrefix(q, kSeq, vSeq, 1.0f, 1024, emptyList()).toDtype(DType.F32)
    val ys = flashAttnVarlenPagedWindowed(
        q, kPaged, vPaged, seqlens, seqlens, blockTable, null,
        seqLen, seqLen, 1.0f, 1024, 0, blockSize, null,
    ).toDtype(DType.F32)

    val diff = abs((ysRef - ys).abs().flattenAll().max(0).toScalarF32())
    assertTrue(diff < 0.125f, "max diff $diff")
}

class FlashAttnTest {
    @Test
    fun flashAttnAcausal() {
        val device = try {
            Device.newCuda(0)
        } catch (_: CandleException) {
            return
        }
        val qRaw = try {
            Tensor.arange(0, 48, device)
        } catch (_: CandleException) {
            return
        }
        val q = qRaw.toDtype(DType.F16).reshape(Shape(1, 3, 2, 8))
        val k = q / 40.0
        val v = q / 50.0
        val qScaled = q / 30.0

        val ys1 = faAcausal(qScaled, k, v, 0.5f)
        val ys1F32 = ys1.narrow(0, 0, 1).squeeze(0).toDtype(DType.F32)
        val ys2 = run {
            val qT = qScaled.transpose(1, 2)
            val kT = k.transpose(1, 2)
            val vT = v.transpose(1, 2)
            flashAttn(qT, kT, vT, 0.5f, false).transpose(1, 2)
        }
        val ys2F32 = ys2.narrow(0, 0, 1).squeeze(0).toDtype(DType.F32)
        val diff = (ys1F32 - ys2F32).abs().flattenAll().max(0)

        assertEquals(listOf(3, 2, 8), ys1F32.dims())
        val expected = listOf(
            listOf(
                listOf(0.0837f, 0.1038f, 0.1238f, 0.1438f, 0.1637f, 0.1837f, 0.2037f, 0.2238f),
                listOf(0.0922f, 0.1122f, 0.1322f, 0.1522f, 0.1721f, 0.1921f, 0.2122f, 0.2322f),
            ),
            listOf(
                listOf(0.4204f, 0.4404f, 0.4604f, 0.4805f, 0.5005f, 0.5205f, 0.5405f, 0.5605f),
                listOf(0.4280f, 0.4480f, 0.4680f, 0.4880f, 0.5083f, 0.5283f, 0.5483f, 0.5684f),
            ),
            listOf(
                listOf(0.7554f, 0.7754f, 0.7954f, 0.8154f, 0.8354f, 0.8555f, 0.8755f, 0.8955f),
                listOf(0.7622f, 0.7822f, 0.8022f, 0.8223f, 0.8423f, 0.8623f, 0.8823f, 0.9023f),
            ),
        )
        assertEquals(expected, toVec3Round(ys1F32, 4))
        assertEquals(listOf(3, 2, 8), ys2F32.dims())
        assertEquals(expected, toVec3Round(ys2F32, 4))
        assertTrue(abs(diff.toScalarF32()) < 1e-5f)
    }

    @Test
    fun flashAttnAcausalSoftcap() {
        val device = try {
            Device.newCuda(0)
        } catch (_: CandleException) {
            return
        }
        val qRaw = try {
            Tensor.arange(0, 3 * 5 * 8, device)
        } catch (_: CandleException) {
            return
        }
        val q = qRaw.toDtype(DType.F16).reshape(Shape(1, 3, 5, 8))
        val k = q / 40.0
        val v = q / 50.0
        val qScaled = q / 30.0
        val softcap = 5.0f

        val ys1 = faAcausalSoftcap(qScaled, k, v, softcap)
        val ys1F32 = ys1.narrow(0, 0, 1).squeeze(0).toDtype(DType.F32)
        val ys2 = run {
            val qT = qScaled.transpose(1, 2)
            val kT = k.transpose(1, 2)
            val vT = v.transpose(1, 2)
            flashAttnAlibiWindowedSoftcap(
                qT,
                kT,
                vT,
                null,
                1.0f,
                null,
                null,
                softcap,
            ).transpose(1, 2)
        }
        val ys2F32 = ys2.narrow(0, 0, 1).squeeze(0).toDtype(DType.F32)
        val diff = (ys1F32 - ys2F32).abs().flattenAll().max(0)

        assertEquals(listOf(3, 5, 8), ys1F32.dims())
        assertEquals(listOf(3, 5, 8), ys2F32.dims())
        assertTrue(abs(diff.toScalarF32()) < 1e-3f)
    }

    @Test
    fun flashAttnVarlenPagedMmPrefixWindowed() {
        val device = try {
            Device.newCuda(0)
        } catch (_: CandleException) {
            return
        }
        val seqLen = 296
        val nHeads = 8
        val nKvHeads = 2
        val headDim = 256
        val blockSize = 32
        val numBlocks = (seqLen + blockSize - 1) / blockSize
        val paddedLen = numBlocks * blockSize
        val elemCount = seqLen * nHeads * headDim
        val qData = (0 until elemCount).map { i -> ((i % 251).toFloat() - 125.0f) / 125.0f }
        val q = try {
            Tensor.fromVec(qData, Shape(seqLen, nHeads, headDim), device).toDtype(DType.BF16)
        } catch (_: CandleException) {
            return
        }
        val kvElemCount = seqLen * nKvHeads * headDim
        val kvData = (0 until kvElemCount).map { i -> ((i % 193).toFloat() - 96.0f) / 96.0f }
        val kv = Tensor.fromVec(kvData, Shape(seqLen, nKvHeads, headDim), device).toDtype(DType.BF16)
        val kSeq = kv / 4.0
        val vSeq = kv / 5.0
        val pad = Tensor.zeros(Shape(paddedLen - seqLen, nKvHeads, headDim), DType.BF16, device)
        val kPaged = Tensor.cat(listOf(kSeq, pad), 0).reshape(Shape(numBlocks, blockSize, nKvHeads, headDim))
        val vPaged = Tensor.cat(listOf(vSeq, pad), 0).reshape(Shape(numBlocks, blockSize, nKvHeads, headDim))
        val seqlens = Tensor.new(uintArrayOf(0u, seqLen.toUInt()), Shape(2), device)
        val blockTable = Tensor.new((0 until numBlocks).map { it.toUInt() }.toUIntArray(), Shape(numBlocks), device)
            .reshape(Shape(1, numBlocks))
        val mmPrefixRanges = Tensor.new(intArrayOf(6, 262), Shape(2), device).reshape(Shape(1, 1, 2))

        val ysRef = faWindowedMmPrefix(q, kSeq, vSeq, 1.0f, 1024, listOf(Pair(6, 262))).toDtype(DType.F32)
        val ysCausalRef = faWindowedMmPrefix(q, kSeq, vSeq, 1.0f, 1024, emptyList()).toDtype(DType.F32)
        val ysCausal = flashAttnVarlenPagedWindowed(
            q,
            kPaged,
            vPaged,
            seqlens,
            seqlens,
            blockTable,
            null,
            seqLen,
            seqLen,
            1.0f,
            1024,
            0,
            blockSize,
            null,
        ).toDtype(DType.F32)
        val causalDiff = abs((ysCausalRef - ysCausal).abs().flattenAll().max(0).toScalarF32())
        assertTrue(causalDiff < 0.125f, "causal max diff $causalDiff")

        val ys = flashAttnVarlenPagedWindowed(
            q,
            kPaged,
            vPaged,
            seqlens,
            seqlens,
            blockTable,
            mmPrefixRanges,
            seqLen,
            seqLen,
            1.0f,
            1024,
            0,
            blockSize,
            null,
        ).toDtype(DType.F32)

        val diff = abs((ysRef - ys).abs().flattenAll().max(0).toScalarF32())
        assertTrue(diff < 0.125f, "max diff $diff")
    }

    @Test
    fun flashAttnVarlenPagedShuffledBlockTableHd256() {
        pagedShuffledBlockTable(256)
    }

    @Test
    fun flashAttnVarlenPagedShuffledBlockTableHd512() {
        pagedShuffledBlockTable(512)
    }

    @Test
    fun flashAttnVarlen() {
        val device = try {
            Device.newCuda(0)
        } catch (_: CandleException) {
            return
        }
        val qRaw = try {
            Tensor.arange(0, 48, device)
        } catch (_: CandleException) {
            return
        }
        val q = qRaw.toDtype(DType.F16).reshape(Shape(3, 2, 8))
        val k = q / 40.0
        val v = q / 50.0
        val qScaled = q / 30.0

        val seqlensQ = Tensor.new(uintArrayOf(0u, 2u), Shape(2), device)
        val seqlensK = Tensor.new(uintArrayOf(0u, 2u), Shape(2), device)

        val ys = run {
            val qT = qScaled.transpose(0, 1)
            val kT = k.transpose(0, 1)
            val vT = v.transpose(0, 1)
            flashAttnVarlen(
                qT,
                kT,
                vT,
                seqlensQ,
                seqlensK,
                32,
                32,
                0.5f,
                false,
            ).transpose(0, 1)
        }.toDtype(DType.F32)

        assertEquals(listOf(3, 2, 8), ys.dims())
        val expected = listOf(
            listOf(
                listOf(0.0837f, 0.1038f, 0.1238f, 0.1438f, 0.1637f, 0.1837f, 0.2037f, 0.2238f),
                listOf(0.0922f, 0.1122f, 0.1322f, 0.1522f, 0.1721f, 0.1921f, 0.2122f, 0.2322f),
            ),
            listOf(
                listOf(0.4204f, 0.4404f, 0.4604f, 0.4805f, 0.5005f, 0.5205f, 0.5405f, 0.5605f),
                listOf(0.4280f, 0.4480f, 0.4680f, 0.4880f, 0.5083f, 0.5283f, 0.5483f, 0.5684f),
            ),
            listOf(
                listOf(0.7554f, 0.7754f, 0.7954f, 0.8154f, 0.8354f, 0.8555f, 0.8755f, 0.8955f),
                listOf(0.7622f, 0.7822f, 0.8022f, 0.8223f, 0.8423f, 0.8623f, 0.8823f, 0.9023f),
            ),
        )
        assertEquals(expected, toVec3Round(ys, 4))
    }
}
