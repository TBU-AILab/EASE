package games.planetwars.core

import util.Vec2d
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*

class ObservationFactoryTest {

    private fun createSampleGameState(): GameState {
        val planet = Planet(
            owner = Player.Player1,
            nShips = 50.0,
            position = Vec2d(100.0, 200.0),
            growthRate = 1.5,
            radius = 20.0,
            transporter = Transporter(
                s = Vec2d(50.0, 50.0),
                v = Vec2d(1.0, 1.0),
                owner = Player.Player1,
                sourceIndex = 0,
                destinationIndex = 1,
                nShips = 10.0
            ),
            id = 1
        )
        return GameState(planets = listOf(planet), gameTick = 0)
    }

    @Test
    fun `test observation includes full data for owned planets`() {
        val gameState = createSampleGameState()
        val observation = ObservationFactory.create(gameState, setOf(Player.Player1))

        assertEquals(1, observation.observedPlanets.size)
        val observedPlanet = observation.observedPlanets[0]
        assertEquals(Player.Player1, observedPlanet.owner)
        assertEquals(50.0, observedPlanet.nShips)
        assertNotNull(observedPlanet.transporter)
        assertEquals(10.0, observedPlanet.transporter?.nShips)
    }

    @Test
    fun `test observation hides ship count and transporter details for non-owners`() {
        val gameState = createSampleGameState()
        val observation = ObservationFactory.create(gameState, setOf(Player.Player2), includeTransporterLocations = false)

        assertEquals(1, observation.observedPlanets.size)
        val observedPlanet = observation.observedPlanets[0]
        assertEquals(Player.Player1, observedPlanet.owner)
        assertNull(observedPlanet.nShips)
        assertNull(observedPlanet.transporter)
    }

    @Test
    fun `test transporter location visibility when enabled`() {
        val gameState = createSampleGameState()
        val observation = ObservationFactory.create(gameState, setOf(Player.Player2), includeTransporterLocations = true)

        assertEquals(1, observation.observedPlanets.size)
        val observedPlanet = observation.observedPlanets[0]
        assertNotNull(observedPlanet.transporter)
        assertNull(observedPlanet.transporter?.nShips)
    }

    @Test
    fun `test transporter location is hidden when disabled`() {
        val gameState = createSampleGameState()
        val observation = ObservationFactory.create(gameState, setOf(Player.Player2), includeTransporterLocations = false)

        assertEquals(1, observation.observedPlanets.size)
        val observedPlanet = observation.observedPlanets[0]
        assertNull(observedPlanet.transporter)
    }
}
