package games.planetwars.agents.random

import games.planetwars.agents.Action
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.*

import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking

/*
* This agent simulates a slow player by adding a delay of delayMillis milliseconds before returning an action.
* Only use this agent for testing the coroutines and the timeout mechanism.
 */

class SlowRandomAgent(val delayMillis: Long = 1000) : PlanetWarsPlayer() {

    override fun getAction(gameState: GameState): Action {
        // Run delayMillis in a coroutine context to avoid blocking the main thread
        runBlocking {
            delay(delayMillis)
        }

        // Logic similar to BetterRandomAgent
        val myPlanets = gameState.planets.filter { it.owner == player && it.transporter == null }
        if (myPlanets.isEmpty()) {
            return Action.doNothing()
        }

        val otherPlanets = gameState.planets.filter { it.owner == player.opponent() || it.owner == Player.Neutral }
        if (otherPlanets.isEmpty()) {
            return Action.doNothing()
        }

        val source = myPlanets.random()
        val target = otherPlanets.random()
        return Action(player, source.id, target.id, source.nShips / 2)
    }

    override fun getAgentType(): String {
        return "Slow Random Agent (delayMillis: ${delayMillis}ms)"
    }
}

fun main() {
    val gameState = GameStateFactory(GameParams()).createGame()
    val agent = SlowRandomAgent()
    agent.prepareToPlayAs(Player.Player1, GameParams())
    val action = agent.getAction(gameState)
    println(action)
}