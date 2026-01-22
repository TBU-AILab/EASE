package util

import java.security.MessageDigest

class HashBasedRNG(seed: ByteArray) {
    private var state: ByteArray = sha256(seed)
    private var counter: Long = 0

    fun next(): Int {
        val counterBytes = counter.toBytes()
        state = sha256(state + counterBytes)
        counter++
        // Return a pseudorandom number (use first 4 bytes)
        return state.sliceArray(0..3).toInt()
    }

    fun nextFloat(): Float {
        return next().toUInt().toFloat() / 0xFFFFFFFF.toFloat()
    }

    private fun sha256(input: ByteArray): ByteArray {
        val digest = MessageDigest.getInstance("SHA-256")
        return digest.digest(input)
    }

    private fun Long.toBytes(): ByteArray {
        return ByteArray(8) { i -> ((this shr (56 - i * 8)) and 0xFF).toByte() }
    }

    private fun ByteArray.toInt(): Int {
        return this.fold(0) { acc, byte -> (acc shl 8) or (byte.toInt() and 0xFF) }
    }
}

// Example usage
fun main() {
    val seed = "some_random_seed".toByteArray()
    val prng = HashBasedRNG(seed)
    val t = System.currentTimeMillis()
    var total = 0.0
    val n = 1_000_000
    repeat(n) {
        total += prng.nextFloat()
    }
    val mean = total / n
    println("Mean: $mean")
    println("Elapsed time: ${System.currentTimeMillis() - t} ms")
}
