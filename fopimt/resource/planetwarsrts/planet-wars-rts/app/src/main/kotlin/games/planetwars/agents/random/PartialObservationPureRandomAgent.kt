package games.planetwars.agents.random

import games.planetwars.agents.Action
import games.planetwars.agents.PartialObservationPlayer
import games.planetwars.core.DefaultHiddenInfoSampler
import games.planetwars.core.GameStateReconstructor
import games.planetwars.core.Observation

class PartialObservationPureRandomAgent() : PartialObservationPlayer() {

    private val sampler = DefaultHiddenInfoSampler(params)
    private val reconstructor = GameStateReconstructor(sampler)

    override fun getAction(observation: Observation): Action {
        // illustrate use of reconstructed game state from observation
        val gameState = reconstructor.reconstruct(observation)
        val source = gameState.planets.random()
        val target = gameState.planets.random()
        return Action(player, source.id, target.id, source.nShips / 2)
    }

    override fun getAgentType(): String {
        return "Pure Random Agent"
    }
}
