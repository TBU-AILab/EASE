package games.planetwars.agents

import games.planetwars.core.Player
import json_rmi.RemoteConstructable
import kotlinx.serialization.Serializable

interface AbstractGameState {
    fun copy(): AbstractGameState
    fun isTerminal(): Boolean
    fun getScore(): Map<Player,Double>
    fun getLegalActions(player: Player): List<Action>
    fun next(actions: Map<Player, Action>): AbstractGameState
}

@Serializable
data class Action (
    val playerId: Player, // the player that is making the move, set to Neutral to do nothing
    val sourcePlanetId: Int,
    val destinationPlanetId: Int,
    val numShips: Double
) : RemoteConstructable {
    companion object {
        val DO_NOTHING = Action(
            playerId = Player.Neutral,
            sourcePlanetId = -1,
            destinationPlanetId = -1,
            numShips = 0.0
        )

        fun doNothing(): Action = DO_NOTHING
    }
}
