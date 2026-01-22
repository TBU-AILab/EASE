package games.planetwars.view

import games.planetwars.core.Player
import xkg.gui.XColor

data class ColorScheme(
    val background: XColor = XColor.black,
    val neutral: XColor = XColor.gray,
    val playerOne: XColor = XColor.red,
    val playerTwo: XColor = XColor.blue,
    val text: XColor = XColor.white,
    val planetOutline: XColor = XColor.gray,
    val transporterOutline: XColor = XColor.white,
) {

    fun getColor(player: Player): XColor {
        return when (player) {
            Player.Player1 -> playerOne
            Player.Player2 -> playerTwo
            Player.Neutral -> neutral
        }
    }
}