package util

import kotlinx.serialization.Serializable
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

@Serializable
data class Vec2d(val x: Double=0.0, val y: Double=0.0) {

    operator fun plus(v: Vec2d) = Vec2d(x + v.x, y + v.y)

    operator fun times(scalar: Double) = Vec2d(x * scalar, y * scalar)

    operator fun minus(v: Vec2d) = Vec2d(x - v.x, y - v.y)

    fun dot(v: Vec2d) = x * v.x + y * v.y

    fun wAdd(v: Vec2d, scalar: Double) = Vec2d(x + v.x * scalar, y + v.y * scalar)

    fun mag() = sqrt(x * x + y * y)

    fun distance(v: Vec2d) = (this - v).mag()

    fun angle() = Math.atan2(y, x)

    fun rotate(angle: Double): Vec2d {
        return Vec2d(
            this.x * cos(angle) - this.y * sin(angle),
            this.x * sin(angle) + this.y * cos(angle)
        )
    }

    // convenience for legacy code
    fun rotatedBy(theta: Double) = rotate(theta)
    fun normalize(): Vec2d {
        val m = mag()
        return if (m > 0.0) this * (1.0 / m) else this
    }

}

fun main() {
    val v1 = Vec2d(3.0, 4.0)
    val v2 = Vec2d(1.0, 2.0)
    println(v1 + v2)
    println(v1.mag())
}

