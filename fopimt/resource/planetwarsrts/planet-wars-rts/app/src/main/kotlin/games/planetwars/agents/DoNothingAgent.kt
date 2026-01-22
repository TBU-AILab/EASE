package games.planetwars.agents

import games.planetwars.core.GameState

class DoNothingAgent : PlanetWarsPlayer() {
    override fun getAction(gameState: GameState): Action {
        return Action.DO_NOTHING
    }

    override fun getAgentType(): String {
        return "DoNothingAgent"
    }
}