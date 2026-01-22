package xkg.examples


import util.Vec2d
import xkg.gui.*
import xkg.jvm.AppLauncher
import kotlin.math.PI

class HelloXGraphics : XApp {

    var x: Int = 0
    var hexAngle = 0.0
    var angleInc = PI / 180

    override fun paint(xg: XGraphics) {
        // draw a rectangle
        val centre = Vec2d(xg.width() / 2, xg.height() / 2)

        val rect = XRect(centre, xg.width() / 2, xg.height() / 2, XStyle())

        val bgRect = rect.copy()
        bgRect.dStyle = XStyle(fg = XColor.magenta, lineWidth = 10.0)
        bgRect.w = xg.width()
        bgRect.h = xg.height()

        xg.draw(bgRect)
        xg.draw(rect)

        val thickWhite = XStyle(lc = XColor.white, lineWidth = 10.0)
        // test line drawing
        xg.draw(XLine(Vec2d(0.0, 0.0), Vec2d(xg.width(), xg.height()), thickWhite))
        xg.draw(XLine(Vec2d(0.0, xg.height()), Vec2d(xg.width(), 0.0), thickWhite))

        hexAngle += angleInc
        val points = PolyUtil().makePolygon(rad = xg.height() / 5, startAngle = hexAngle)
        val poly = XPoly(Vec2d(xg.width() / 2, xg.height() / 2), points)
        with(poly.dStyle) {
            fg = XColor(r = 0.5f, g = 1f)
            lc = XColor(r = 0.5f, b = 1f)
            lineWidth = 5.0
            fill = fillHex
        }
        xg.draw(poly)

        val roundedRect = XRoundedRect(
            Vec2d(xg.width() / 8, xg.height() / 8),
            xg.width() / 4, xg.height() / 4, cornerRad = 50.0, radInPercent = false,
            rotation = hexAngle
        )
        xg.draw(roundedRect)

        // val message = Character
        val text = XText("Hello XKG", centre, TStyle(fg = XColor.black), XStyle(), rotation = hexAngle)
        xg.draw(text)

    }

    override fun handleMouseEvent(e: XMouseEvent) {

    }

    var fillHex = true
    override fun handleKeyEvent(e: XKeyEvent) {
        // println(e)
        if (e.t == XKeyEventType.Pressed) fillHex = !fillHex
    }

}

fun main() {
    AppLauncher(HelloXGraphics()).launch()
}
