package competition_entry

import games.planetwars.agents.Action
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.GameState
import games.planetwars.core.Player
import games.planetwars.core.Planet
import kotlin.math.max

/*
    * Greedy Heuristic Agent for Planet Wars
    * This agent was written by ChatGPT 4o, and is intended to be a simple heuristic-based agent
 */

class GreedyHeuristicAgent : PlanetWarsPlayer() {
    override fun getAction(gameState: GameState): Action {
        // Planets owned by us with no current transporter
        val myPlanets = gameState.planets.filter { it.owner == player && it.transporter == null && it.nShips > 10 }
        if (myPlanets.isEmpty()) return Action.doNothing()

        // Planets not owned by us
        val candidateTargets = gameState.planets.filter { it.owner != player }

        // Choose best source planet (most ships)
        val source = myPlanets.maxByOrNull { it.nShips } ?: return Action.doNothing()

        // Find the best target using a heuristic
        val target = candidateTargets.minByOrNull { target ->
            val distance = source.position.distance(target.position)
            val shipStrength = if (target.owner == Player.Neutral) target.nShips else target.nShips * 1.5
            val score = shipStrength + distance - 2 * target.growthRate
            score
        } ?: return Action.doNothing()

        // Only attack if we have enough ships to make a difference
        val estimatedDefense =
            target.nShips + target.growthRate * (source.position.distance(target.position) / params.transporterSpeed)
        if (source.nShips <= estimatedDefense) return Action.doNothing()

        // Send half our ships
        val shipsToSend = source.nShips / 2
        return Action(player, source.id, target.id, shipsToSend)
    }

    override fun getAgentType(): String = "Greedy Heuristic Agent"
}
