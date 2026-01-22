package games.planetwars.agents.random

import games.planetwars.agents.Action
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.GameParams
import games.planetwars.core.GameState
import games.planetwars.core.GameStateFactory
import games.planetwars.core.Player

class CarefulRandomAgent() : PlanetWarsPlayer() {
    override fun getAction(gameState: GameState): Action {
        // filter the planets that are owned by the player AND have a transporter available
        val myPlanets = gameState.planets.filter { it.owner == player && it.transporter == null }
        if (myPlanets.isEmpty()) {
            return Action.doNothing()
        }
        // now find a random target planet owned by the opponent
        val opponentPlanets = gameState.planets.filter { it.owner == player.opponent() }
        if (opponentPlanets.isEmpty()) {
            return Action.doNothing()
        }
        val source = myPlanets.random()
        val target = opponentPlanets.random()
        return Action(player, source.id, target.id, source.nShips/2)
    }

    override fun getAgentType(): String {
        return "Careful Random Agent"
    }
}

fun main() {
    val agent = CarefulRandomAgent()
    agent.prepareToPlayAs(Player.Player1, GameParams())
    val gameState = GameStateFactory(GameParams()).createGame()
    val action = agent.getAction(gameState)
    println(action)
}
