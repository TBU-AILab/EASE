package xkg.jvm

import util.Vec2d
import xkg.gui.*
import java.awt.*
import java.awt.event.KeyAdapter
import java.awt.event.KeyEvent
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.awt.geom.*
import javax.swing.JComponent
import kotlin.math.min

class XGraphicsJVMOld(val jc: JComponent) : XGraphics {

    val drawables = ArrayList<Drawable>()

    // var transform = Transform

    var rect: LRect? = null
    override fun setBounds(rect: LRect) {
        this.rect = rect

    }

    override fun releaseBounds() {
        rect = null
    }

    override fun setTranslate(x: Double, y: Double) {
        graphics2D?.let { it.translate(x, y) }
    }

    override fun setScale(x: Double, y: Double) {
        graphics2D?.let { it.scale(x, y) }
    }

    override fun width(): Double {
        val bounds = rect
        return if (bounds != null) bounds.width else jc.width.toDouble()
    }

    override fun height(): Double {
        val bounds = rect
        return if (bounds != null) bounds.height else jc.height.toDouble()
    }

    var graphics2D: Graphics2D? = null

    override fun draw(toDraw: Drawable) {
        val g = graphics2D
        if (g != null) {
            // apply the translation if necessary

            val bounds = rect
            if (bounds != null) {
                g.translate(bounds.xLeft, bounds.yTop)
                g.setClip(0, 0, width().toInt(), height().toInt())
            }

            if (toDraw is XRect) drawRect(toDraw)
            if (toDraw is XRoundedRect) drawRoundedRect(toDraw)
            if (toDraw is XEllipse) drawEllipse(toDraw)
            if (toDraw is XPoly) drawPoly(toDraw)
            if (toDraw is XLine) drawLine(toDraw)
            if (toDraw is XQuadCurve) drawQuadCurve(toDraw)
            if (toDraw is XText) drawText(toDraw)

            if (bounds != null) {
                g.clip = null
                g.translate(-bounds.xLeft, -bounds.yTop)
            }


        }
    }

    fun getColor(c: XColor): Color {
        return Color(c.r, c.g, c.b, c.a)
    }

    fun drawRect(rect: XRect) {
        val g = graphics2D
        if (g != null) {
            with(rect) {
                val at = g.transform
                g.translate(centre.x, centre.y)
                g.rotate(rotation)
                val r2d = Rectangle2D.Double(-w / 2, -h / 2, w, h)
                with(rect.dStyle) {
                    if (fill) {
                        g.color = getColor(fg)
                        g.fill(r2d)
                    }
                    if (stroke) {
                        g.color = getColor(lc) // Color(lc.r.toFloat(), lc.g.toFloat(), lc.b.toFloat())
                        g.stroke = BasicStroke(lineWidth.toFloat())
                        g.draw(r2d)
                    }
                }
                g.transform = at
            }
        }
    }

    fun drawRoundedRect(rect: XRoundedRect) {
        val g = graphics2D
        if (g != null) {
            with(rect) {
                val at = g.transform
                g.translate(centre.x, centre.y)
                g.rotate(rotation)
                val arc =
                    if (rect.radInPercent) min(w, h) * cornerRad
                    else cornerRad
                val r2d = RoundRectangle2D.Double(-w / 2, -h / 2, w, h, arc, arc)
                with(rect.dStyle) {
                    if (fill) {
                        g.color = getColor(fg)
                        g.fill(r2d)
                    }
                    if (stroke) {
                        g.color = getColor(lc) // Color(lc.r.toFloat(), lc.g.toFloat(), lc.b.toFloat())
                        g.stroke = BasicStroke(lineWidth.toFloat())
                        g.draw(r2d)
                    }
                }
                g.transform = at
            }
        }
    }

    fun drawEllipse(ellipse: XEllipse) {
        val g = graphics2D
        if (g != null) {
            with(ellipse) {
                val at = g.transform
                g.translate(centre.x, centre.y)
                g.rotate(rotation)
                val r2d = Ellipse2D.Double(-w / 2, -h / 2, w, h)
                with(ellipse.dStyle) {
                    if (fill) {
                        g.color = getColor(fg)
                        g.fill(r2d)
                    }
                    if (stroke) {
                        g.color = getColor(lc) // Color(lc.r.toFloat(), lc.g.toFloat(), lc.b.toFloat())
                        g.stroke = BasicStroke(lineWidth.toFloat())
                        g.draw(r2d)
                    }
                }
                g.transform = at
            }
        }
    }

    fun drawLine(line: XLine) {
        val g = graphics2D
        if (g != null) {
            with(line) {
                with(line.dStyle) {
                    if (stroke) {
                        val at = g.transform
                        val p = (a + b) * 0.5
                        val c = a - p
                        val d = b - p
                        val l2d = Line2D.Double(c.x, c.y, d.x, d.y)
                        g.translate(p.x, p.y)
                        g.rotate(rotation)
                        g.color = getColor(lc)
                        g.stroke = BasicStroke(lineWidth.toFloat())
                        g.draw(l2d)
                        g.transform = at
                    }
                }
            }
        }
    }

    private fun drawQuadCurve(quadCurve: XQuadCurve) {
        val g = graphics2D
        if (g != null) {
            with(quadCurve) {
                val q2d = java.awt.geom.QuadCurve2D.Double(a.x, a.y, b.x, b.y, c.x, c.y)
                with(dStyle) {
                    if (stroke) {
                        g.color = getColor(lc)
                        g.stroke = BasicStroke(lineWidth.toFloat())
                        g.draw(q2d)
                    }
                }
            }
        }
    }


    fun drawText(text: XText) {
        val g = graphics2D
        if (g != null) {
            with(text) {
                with(tStyle) {
                    g.color = getColor(fg)
                    g.setFont(Font("Monospaced", Font.BOLD, size.toInt()))
                    val rect: Rectangle2D = g.getFontMetrics().getStringBounds(str, g)
                    val sx = -rect.width.toInt() / 2
                    val sy = (rect.height.toInt() / 2 - g.getFontMetrics().getDescent())

                    val at = g.transform
                    g.translate(p.x, p.y)
                    g.rotate(rotation)
                    if (flipped) {
                        g.scale(1.0, -1.0)
                    }
                    g.drawString(str, sx, sy)
                    g.transform = at
                }
            }
        }
        // println(text)
    }


    fun drawPoly(poly: XPoly) {
        if (poly.points.isEmpty()) return
        val path = Path2D.Double()
        // path.contains(0.0, 0.0)
        with(poly) {
            path.moveTo(points[0].x, points[0].y)
            for (v in points) path.lineTo(v.x, v.y)
            if (poly.closed) path.closePath()
        }
        val g = graphics2D
        if (g != null) {
            with(poly.dStyle) {
                val at = g.transform
                g.translate(poly.centre.x, poly.centre.y)
                g.rotate(poly.rotation)
                if (fill) {
                    g.color = getColor(fg)
                    g.fill(path)
                }
                if (stroke) {
                    g.color = getColor(lc)
                    g.stroke = BasicStroke(lineWidth.toFloat())
                    g.draw(path)
                }
                g.transform = at
            }
        }
    }

    override fun drawables(): ArrayList<Drawable> {
        return drawables
    }

    override fun redraw() {
        // may be not needed
        // for ()
        println("Should redraw, but what?")
    }

    override var style: XStyle = XStyle()
}
