package games.planetwars.core

import util.Vec2d
import kotlin.random.Random
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*

class GameStateReconstructorTest {

    private val params = GameParams()
    private val sampler = DefaultHiddenInfoSampler(params)
    private val reconstructor = GameStateReconstructor(sampler)

    @Test
    fun `test reconstructed game state has correct known values`() {
        val observedPlanet = PlanetObservation(
            owner = Player.Player1,
            nShips = 25.0, // Known ship count
            position = Vec2d(100.0, 200.0),
            growthRate = 1.5,
            radius = 20.0,
            transporter = null, // No transporter
            id = 1
        )
        val observation = Observation(listOf(observedPlanet), gameTick = 10)
        val reconstructedGameState = reconstructor.reconstruct(observation)

        assertEquals(1, reconstructedGameState.planets.size)
        val reconstructedPlanet = reconstructedGameState.planets[0]
        assertEquals(Player.Player1, reconstructedPlanet.owner)
        assertEquals(25.0, reconstructedPlanet.nShips)
        assertNull(reconstructedPlanet.transporter)
    }

    @Test
    fun `test unknown ship counts are sampled`() {
        val observedPlanet = PlanetObservation(
            owner = Player.Player2,
            nShips = null, // Unknown ship count
            position = Vec2d(150.0, 250.0),
            growthRate = 1.2,
            radius = 18.0,
            transporter = null, // No transporter
            id = 2
        )
        val observation = Observation(listOf(observedPlanet), gameTick = 15)
        val reconstructedGameState = reconstructor.reconstruct(observation)

        assertNotNull(reconstructedGameState.planets[0].nShips)
    }

    @Test
    fun `test transporter ship counts are sampled when unknown`() {
        val observedPlanet = PlanetObservation(
            owner = Player.Player1,
            nShips = 30.0,
            position = Vec2d(200.0, 300.0),
            growthRate = 1.8,
            radius = 22.0,
            transporter = TransporterObservation(
                s = Vec2d(50.0, 50.0),
                v = Vec2d(1.0, 1.0),
                owner = Player.Player1,
                sourceIndex = 0,
                destinationIndex = 1,
                nShips = null // Unknown transporter ship count
            ),
            id = 3
        )
        val observation = Observation(listOf(observedPlanet), gameTick = 20)
        val reconstructedGameState = reconstructor.reconstruct(observation)

        assertNotNull(reconstructedGameState.planets[0].transporter)
        assertNotNull(reconstructedGameState.planets[0].transporter?.nShips)
    }
}
