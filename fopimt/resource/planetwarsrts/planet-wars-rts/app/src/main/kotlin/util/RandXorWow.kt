package util

class RandXorWow (seed: Int? = null) {
    private val state = IntArray(5)
    private var v = 0xFFFFFFFF.toInt()

    init {
        val initialSeed = seed ?: System.currentTimeMillis().toInt()
        state[0] = initialSeed
        for (i in 1 until 5) {
            state[i] = state[i - 1] xor ((state[i - 1] ushr 30) + i)
        }
    }

    fun next(): Int {
        var t = state[4]
        t = t xor (t ushr 2)
        t = t xor (t shl 1)
        t = t xor (t shl 4)
        state[4] = state[3]
        state[3] = state[2]
        state[2] = state[1]
        state[1] = state[0]
        t = t xor state[0]
        t = t xor (state[0] ushr 31)
        state[0] = t

        v += 362437
        return t + v
    }

    fun nextFloat(): Float {
        return next().toUInt().toFloat() / 0xFFFFFFFF.toFloat()
    }
}

fun main() {
    val rand = RandXorWow()
    for (i in 0 until 10) {
//        println(rand.next())
        println(rand.nextFloat())
    }
}
