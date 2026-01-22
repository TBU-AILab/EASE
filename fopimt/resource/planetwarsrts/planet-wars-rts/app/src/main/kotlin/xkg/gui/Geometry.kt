package xkg.gui

import util.Vec2d


class Poly {
    fun contains(p: Vec2d, points: List<Vec2d>): Boolean {
        var result = false
        var j = points.size - 1
        for (i in 0 until points.size) {
            val pi = points[i]
            val pj = points[j]
            if (pi.y < p.y && pj.y >= p.y || pj.y < p.y && pi.y >= p.y) {
                if (pi.x + (p.y - pi.y) / (pj.y - pi.y) * (pj.x - pi.x) < p.x) {
                    result = !result
                }
            }
            j = i
            // if (polygon[i].X + (testPoint.Y - polygon[i].Y) / (polygon[j].Y - polygon[i].Y) *
            // (polygon[j].X - polygon[i].X) < testPoint.X)
        }
        return result
    }
}


enum class Horizontal { Left, Center, Right, Any }
enum class Vertical { Top, Middle, Bottom, Any }
enum class Expansion { Fill, Squash }
enum class Direction {}

// todo: need ana easy way to draw each aapp in the LRect
// in its own rectangle

// todo: look in to best ways of doing this - it's very simple
// in essence, just need to translate the graphics transform
// for each one, set a clippng region, and then call the paint
// method of the child xApp

// obviously this can be done recursively also - just need to add the children
// to the LRect structure

data class LRect(
    var xLeft: Double = 0.0, var yTop: Double = 0.0,
    var width: Double, var height: Double,
    var app: XApp? = null
) {
    fun centre() = Vec2d(xLeft + width / 2, yTop + height / 2)
    fun XRect(style: XStyle) = XRect(centre(), width, height, style)
}

data class Span(val from: Double, val to: Double) {
    fun size() = to - from
}

// if aspect ratio is null then don't care about aspect ratio
// padding is for margin and between cells
data class LPane(
    var rect: LRect,
    val aspectRatio: Double? = null,
    val hPos: Horizontal = Horizontal.Any,
    val vPos: Vertical = Vertical.Any,
    val hFill: Expansion = Expansion.Fill,
    val vFill: Expansion = Expansion.Fill,
    val padding: Double = 0.02
)

class Layout(var padding: Double = 0.02) {

    fun doLayout() {

    }


    fun spaceEvenly(n: Int): List<LRect> {

        val panes = ArrayList<LRect>()
        val padTotal = padding * (n + 1)

        val remaining = 1.0 - padTotal

        return panes
    }

    fun hPartition(w: Double, h: Double, n: Int, ratios: ArrayList<Double> = ArrayList()): List<LRect> {
        val hSpans = getSpans(w, n, ratios)
        val vSpans = getSpans(h, 1)
        return expand(hSpans, vSpans)
    }

    fun vPartition(w: Double, h: Double, n: Int, ratios: ArrayList<Double> = ArrayList()): List<LRect> {
        val hSpans = getSpans(w, 1)
        val vSpans = getSpans(h, n, ratios)
        return expand(hSpans, vSpans)
    }

    fun expand(hSpans: List<Span>, vSpans: List<Span>): List<LRect> {
        val panes = ArrayList<LRect>()
        for (h in hSpans) {
            for (v in vSpans) {
                panes.add(LRect(h.from, v.from, h.size(), v.size()))
            }
        }
        return panes
    }

    fun getSpans(total: Double, n: Int, ratios: ArrayList<Double> = ArrayList<Double>()): List<Span> {

        if (ratios.isEmpty())
            (1..n).forEach { ratios.add(1.0 / n.toDouble()) }

        val spans = ArrayList<Span>()
        val padTotal = padding * (n + 1)
        val remaining = 1.0 - padTotal

        var cur = 0.0
        for (i in 0 until n) {
            // add padding in each timee
            cur += padding * total

            val sp = ratios[i] * total * remaining
            spans.add(Span(cur, cur + sp))

            cur += sp
        }
        return spans
    }

}



