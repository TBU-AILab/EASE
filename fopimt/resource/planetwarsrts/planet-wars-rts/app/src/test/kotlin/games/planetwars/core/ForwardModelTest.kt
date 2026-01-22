package games.planetwars.core

import games.planetwars.agents.Action
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class ForwardModelTest {

    @Test
    fun `test transporter sends and arrives correctly`() {

        val params = GameParams(minInitialShipsPerPlanet = 10, numPlanets = 2, initialNeutralRatio = 0.0)
        val state = GameStateFactory(params).createGame()
        val model = ForwardModel(state, params)


        val source = state.deepCopy().planets.firstOrNull { it.owner == Player.Player1 && it.nShips >= 10 }
            ?: error("No suitable source planet found for Player1")

        val target = state.deepCopy().planets.firstOrNull { it.owner == Player.Player2 }
            ?: error("No suitable target planet found for Player2")


        val initialSourceShips = source.nShips

        val action = Action(Player.Player1, source.id, target.id, 10.0)
        println("Action: $action")

        val nShips = action.numShips

        // Step 1: send the transporter
        model.step(mapOf(Player.Player1 to action))

        println(state)

        // Check source ships decreased and transporter created
        assertEquals(initialSourceShips - nShips, state.planets[source.id].nShips, 0.1)
        assertNotNull(state.planets[source.id].transporter, "Transporter should have been launched")

        // Step forward until transporter arrives at target
        while (state.planets[source.id].transporter != null) {
            model.step(emptyMap())
        }

        // when the transporter arrives, the target should have received the invading ships
        // and source and target planets should now have the same number
        assertEquals(state.planets[source.id].nShips, state.planets[target.id].nShips, 0.1)

    }
}
