package xkg.gui


import util.Vec2d
import kotlin.math.*
import kotlin.random.Random


data class XColor(var r: Float = 0f, var g: Float = 0f, var b: Float = 0f, var a: Float = 1f) {
    companion object {
        val red = XColor(r = 1f)
        val green = XColor(g = 1f)
        val blue = XColor(b = 1f)
        val magenta = XColor(r = 1f, b = 1f)
        val cyan = XColor(g = 1f, b = 1f)
        val yellow = XColor(r = 1f, g = 1f)
        val white = XColor(r = 1f, g = 1f, b = 1f)
        val black = XColor()
        val gray = XColor(r = 0.5f, g = 0.5f, b = 0.5f)
        val pink = XColor(r = 1f, g = 0.5f, b = 0.5f)
        val purple = XColor(r = 0.5f, b = 0.5f)
        val orange = XColor(r = 1f, g = 0.5f)
        val brown = XColor(r = 0.5f, g = 0.25f)
    }
}

// provide a reusable palette of colors - use RGB space for now,
// but would be better to generate in HSV space to get niccer colours more easily
data class XPalette(
    val nColors: Int = 30, val min: Double = 0.4, val max: Double = 0.9,
    val seed: Long = 1L, val alpha: Float = 1.0f
) {
    val colors = ArrayList<XColor>()
    val rand = if (seed == -1L) Random else Random(seed)

    init {
        for (i in 0 until nColors)
            colors.add(XColor(v(), v(), v(), alpha))
    }

    fun getColor(ix: Int) = colors[ix % colors.size]

    fun v() = rand.nextDouble(min, max).toFloat()
}

class XColorHeat {
    fun getColor(x: Double) : XColor {

        // treat zero a bit specially
        if (x == 0.0) return XColor.black

        // clamp x to be between zero and one
        var v = x.toFloat();
        if (v < 0.0) v = 0f
        if (v > 1.0) v = 1f
        val r = max(0f, 2 * (v - 0.5f))
        val b = max(2*(0.5f - v), 0f)
        // val g = 0.5f // v/4 // 1f - (r+b)
        val g = 1f - (r+b)
        return XColor(r, g, b)
    }
}

data class OldXColor(var r: Double = 0.0, var g: Double = 0.0, var b: Double = 0.0, var a: Double = 1.0)


interface XGraphics {
    fun width(): Double
    fun height(): Double
    fun draw(toDraw: Drawable)
    fun drawables(): ArrayList<Drawable>
    fun redraw()
    var style: XStyle
    fun centre() = Vec2d(width() / 2, height() / 2)

    // do nothing by default to not break anything
    fun setBounds(rect: LRect)
    fun releaseBounds()

    fun saveTransform() {}
    fun restoreTransform() {}

    fun setTranslate(x: Double, y: Double) {}

    // var pane: LRect
    fun setScale(x: Double, y: Double)
}


//abstract class Rotatable (var rotation: Double = 0.0) : Drawable {
//
//}


data class XStyle(
    // default background, foreground and line colours
    var fg: XColor = XColor.black,
    var bg: XColor = XColor.white,
    var lc: XColor = XColor.red,
    var stroke: Boolean = true,
    var fill: Boolean = true,
    var lineWidth: Double = 2.0
)

data class TStyle(
    // default background, foreground and line colours
    var fg: XColor = XColor.cyan,
    var font: String = "Arial",
    var size: Double = 16.0,
    var flipped: Boolean = false,
)

interface XApp {
    fun paint(xg: XGraphics) {}
    fun handleMouseEvent(e: XMouseEvent) {}
    fun handleKeyEvent(e: XKeyEvent) {}
}

enum class XMouseEventType { Down, Up, Moved, Dragged, Clicked }
data class XMouseEvent(val t: XMouseEventType, val s: Vec2d)

enum class XKeyEventType { Pressed, Released, Typed, Down }
data class XKeyEvent(val t: XKeyEventType, val keyCode: Int)

class XKeyMap {
    companion object {
        val left = 37
        val up = 38
        val right = 39
        val down = 40
        val space = 32
    }
}


interface Drawable {
    var dStyle: XStyle
    var rotation: Double
}

interface GeomDrawable : Drawable {
    fun contains(p: Vec2d?) : Boolean
    // this radius is just a rough approximation -
    // non circular shapes should return radius of bounding circle
    fun radius() : Double
    var centre: Vec2d

}

data class XRect(
    override var centre: Vec2d, var w: Double, var h: Double,
    override var dStyle: XStyle = XStyle(), override var rotation: Double = 0.0
) : GeomDrawable {
    override fun contains(p: Vec2d?): Boolean {
        if (p == null) return false
        // also need to cope with rotation
        // - so transform the point first, then counter-rotate it

        val tp = (p-centre).rotatedBy(-rotation)
        return abs(tp.x) <= w/2 && abs(tp.y) <= h/2
    }

    override fun radius()= max(w/2,h/2)
}

data class XRoundedRect(
    override var centre: Vec2d, var w: Double, var h: Double,
    var cornerRad:Double = 0.025,
    var radInPercent: Boolean = true,  // specifies whether percent or pixels
    override var dStyle: XStyle = XStyle(), override var rotation: Double = 0.0
) : GeomDrawable {

    // the contains method is not accurate yet -
    // currently it is just for a standard Rectangle, not a rounded one
    override fun contains(p: Vec2d?): Boolean {
        if (p == null) return false
        // also need to cope with rotation
        // - so transform the point first, then counter-rotate it

        val tp = (p-centre).rotatedBy(-rotation)
        return abs(tp.x) <= w/2 && abs(tp.y) <= h/2
    }

    override fun radius()= max(w/2,h/2)
}


data class XEllipse(
    override var centre: Vec2d, var w: Double, var h:
    Double, override var dStyle: XStyle = XStyle(),
    override var rotation: Double = 0.0
) : GeomDrawable {
    val a2 = (w/2) * (w/2)
    val b2 = (h/2) * (h/2)
    override fun contains(p: Vec2d?): Boolean {
        if (p == null) return false
        // also need to cope with rotation
        // - so transform the point first, then counter-rotate it
        val tp = (p-centre).rotatedBy(-rotation)
        return (tp.x*tp.x) / a2 + (tp.y*tp.y) / b2 <= 1
    }
    override fun radius()= max(w/2,h/2)
}

data class XLine(
    var a: Vec2d, var b: Vec2d,
    override var dStyle: XStyle = XStyle(), override var rotation: Double = 0.0
) : Drawable

data class XQuadCurve(
    var a: Vec2d, var b: Vec2d, var c: Vec2d,
    override var dStyle: XStyle = XStyle(), override var rotation: Double = 0.0
) : Drawable

data class XText(
    var str: String, var p: Vec2d, var tStyle: TStyle = TStyle(),
    override var dStyle: XStyle = XStyle(), override var rotation: Double = 0.0
) : Drawable

data class XPoly(
    override var centre: Vec2d = Vec2d(),
    val points: ArrayList<Vec2d> = ArrayList(),
    override var dStyle: XStyle = XStyle(),
    override var rotation: Double = 0.0,
    var closed: Boolean = true
) : GeomDrawable {

    var rad = 0.0
    init { points.forEach { rad = max( rad , it.mag() ) }}

    override fun contains(p: Vec2d?): Boolean {
        if (p == null) return false
        val tp = (p-centre).rotatedBy(-rotation)
        if (tp.mag() > rad) return false
        return Poly().contains(tp, points)
    }

    override fun radius() = rad
}

//data class XPolyOld (var start: Vec2d, val points: ArrayList<Vec2d>) : Drawable {
//    private var intStyle = XStyle()
//    override var dStyle: XStyle
//        get() = intStyle
//        set(value) {intStyle = value}
//}
//
// data class XPolyRegular (val centre: Vec2d, val vRad: Double, val startAngle: Double) : Drawable
class PolyUtil {
    fun makePolygon(n: Int = 6, rad: Double = 10.0, startAngle: Double = 0.0): ArrayList<Vec2d> {
        val verts = ArrayList<Vec2d>()

        val step = (2 * PI) / n
        for (i in 0 until n) {
            val angle = startAngle + i * step
            val x = rad * sin(angle)
            val y = rad * cos(angle)
            verts.add(Vec2d(x, y))
        }
        return verts
    }
}


