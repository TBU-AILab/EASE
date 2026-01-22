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

class EasyComponent(val prefWidth: Int = 600, val prefHeight: Int = 400) : JComponent() {
    override fun getPreferredSize() = Dimension(prefWidth, prefHeight)

    var xg: XGraphicsJVM? = null
    var xApp: XApp? = null

    override fun paintComponent(g: Graphics?) {
        super.paintComponent(g)
        val safeXG = xg
        val app = xApp
        if (safeXG != null && app != null) {
            safeXG.graphics2D = g as Graphics2D
            app.paint(safeXG)
            // safeXG.redraw()
        }
    }

    fun listenForEvents() {
        val xa = xApp
        if (xa != null) addKeyListener(XKeyAdapter(xa))
        println(keyListeners)
    }
}

class XMouseAdapter(val xApp: XApp) : MouseAdapter() {
    override fun mouseClicked(e: MouseEvent?) {
        // super.mouseClicked(e)
        if (e != null)
            xApp.handleMouseEvent(
                XMouseEvent(
                    XMouseEventType.Clicked, Vec2d(e.x.toDouble(), e.y.toDouble())
                )
            )
    }

    override fun mousePressed(e: MouseEvent?) {
        super.mousePressed(e)
        if (e != null)
            xApp.handleMouseEvent(
                XMouseEvent(
                    XMouseEventType.Down, Vec2d(e.x.toDouble(), e.y.toDouble())
                )
            )
    }

    override fun mouseMoved(e: MouseEvent?) {
        super.mouseMoved(e)
        if (e != null)
            xApp.handleMouseEvent(
                XMouseEvent(
                    XMouseEventType.Moved, Vec2d(e.x.toDouble(), e.y.toDouble())
                )
            )
    }

    override fun mouseDragged(e: MouseEvent?) {
        super.mouseDragged(e)
        if (e != null)
            xApp.handleMouseEvent(
                XMouseEvent(
                    XMouseEventType.Dragged, Vec2d(e.x.toDouble(), e.y.toDouble())
                )
            )
    }

    // can do this for the other types of MouseEvent also

}

class XKeyAdapter(val xApp: XApp) : KeyAdapter() {
    override fun keyPressed(e: KeyEvent?) {
        // println(e)
        super.keyPressed(e)
        if (e != null) xApp.handleKeyEvent(XKeyEvent(XKeyEventType.Pressed, e.keyCode))
    }

    override fun keyReleased(e: KeyEvent?) {
        super.keyReleased(e)
        if (e != null) xApp.handleKeyEvent(XKeyEvent(XKeyEventType.Released, e.keyCode))
    }

    override fun keyTyped(e: KeyEvent?) {
        super.keyTyped(e)
        if (e != null) xApp.handleKeyEvent(XKeyEvent(XKeyEventType.Typed, e.keyCode))
    }
}

