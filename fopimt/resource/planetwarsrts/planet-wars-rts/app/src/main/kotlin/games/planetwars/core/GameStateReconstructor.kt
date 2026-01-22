package games.planetwars.core

import util.Vec2d
import kotlin.random.Random

class GameStateReconstructor(private val sampler: HiddenInfoSampler) {
    fun reconstruct(observation: Observation): GameState {
        val reconstructedPlanets = observation.observedPlanets.map { observedPlanet ->
            Planet(
                owner = observedPlanet.owner,
                nShips = observedPlanet.nShips ?: sampler.sampleShips(),
                position = observedPlanet.position,
                growthRate = observedPlanet.growthRate,
                radius = observedPlanet.radius,
                transporter = observedPlanet.transporter?.let { observedTransporter ->
                    Transporter(
                        s = observedTransporter.s,
                        v = observedTransporter.v,
                        owner = observedTransporter.owner,
                        sourceIndex = observedTransporter.sourceIndex,
                        destinationIndex = observedTransporter.destinationIndex,
                        nShips = observedTransporter.nShips ?: sampler.sampleTransporterShips()
                    )
                },
                id = observedPlanet.id
            )
        }
        return GameState(reconstructedPlanets, observation.gameTick)
    }
}

interface HiddenInfoSampler {
    fun sampleShips(): Double
    fun sampleTransporterShips(): Double
}

class DefaultHiddenInfoSampler(private val params: GameParams) : HiddenInfoSampler {
    override fun sampleShips(): Double {
        return Random.nextDouble(params.minInitialShipsPerPlanet.toDouble(), params.maxInitialShipsPerPlanet.toDouble())
    }

    override fun sampleTransporterShips(): Double {
        return Random.nextDouble(1.0, params.maxInitialShipsPerPlanet.toDouble() / 2)
    }
}

fun main() {
    val params = GameParams()
    val sampler = DefaultHiddenInfoSampler(params)
    val reconstructor = GameStateReconstructor(sampler)

    val observedPlanet = PlanetObservation(
        owner = Player.Player1,
        nShips = null, // Unknown ship count
        position = Vec2d(100.0, 200.0),
        growthRate = 1.5,
        radius = 20.0,
        transporter = TransporterObservation(
            s = Vec2d(50.0, 50.0),
            v = Vec2d(1.0, 1.0),
            owner = Player.Player1,
            sourceIndex = 0,
            destinationIndex = 1,
            nShips = null // Unknown transporter ship count
        ),
        id = 1
    )
    val observation = Observation(listOf(observedPlanet), gameTick = 10)
    val reconstructedGameState = reconstructor.reconstruct(observation)
    println(reconstructedGameState)
}
