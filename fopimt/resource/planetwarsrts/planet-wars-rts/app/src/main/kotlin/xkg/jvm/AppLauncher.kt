package xkg.jvm

import xkg.gui.XApp
import java.awt.BorderLayout
import java.awt.Component
import javax.swing.JFrame
import javax.swing.WindowConstants

class JEasyFrame(var comp: Component, title: String) : JFrame(title) {
    init {
        contentPane.add(BorderLayout.CENTER, comp)
        pack()
        this.isVisible = true
        defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
        repaint()
    }
}

class AppLauncher(
    val app: XApp,
    var frameRate: Double = 25.0,
    title: String = "Capture the Flag",
    val preferredWidth: Int = 640,
    val preferredHeight: Int = 480,
) {
    val ec = EasyComponent(preferredWidth, preferredHeight)
    val frame = JEasyFrame(ec, title)
    val xg = XGraphicsJVM(ec)

    init {
        globFrameRate = frameRate
    }

    fun launch() {
        ec.xg = xg
        ec.xApp = app
        frame.addKeyListener(XKeyAdapter(app))
        frame.addMouseListener(XMouseAdapter(app))
        while (true) {
            ec.repaint()
            val delay = 1000 / globFrameRate
//            println("Delay = $delayMillis")
            Thread.sleep(delay.toLong())
        }
    }

    companion object {
        var globFrameRate = 25.0
    }
}
