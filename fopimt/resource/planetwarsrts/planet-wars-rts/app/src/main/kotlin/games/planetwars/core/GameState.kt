package games.planetwars.core

import json_rmi.RemoteConstructable
import kotlinx.serialization.Serializable
import util.Vec2d

// define an enum class for the owner of a planet which could be either
// player 1, player 2, or neutral

@Serializable
enum class Player {
    Player1, Player2, Neutral;

    fun opponent(): Player {
        return when (this) {
            Player1 -> Player2
            Player2 -> Player1
            else -> throw IllegalArgumentException("Neutral does not have an opposite")
        }
    }
}

@Serializable
data class Planet (
    var owner: Player,
    var nShips: Double, // need to support fractional ships for easy calculations
    val position: Vec2d,
    val growthRate: Double,
    val radius: Double,
    var transporter: Transporter? = null, // null means we're free to create one, otherwise it's in transit and not available
    var id: Int = -1,  // will be more convenient to set id later
): RemoteConstructable

@Serializable
data class Transporter (
    var s: Vec2d,
    var v: Vec2d,
    val owner: Player,
    val sourceIndex: Int,
    val destinationIndex: Int,
    val nShips: Double,
) : RemoteConstructable{

}

@Serializable
data class GameState (
    val planets: List<Planet>,  // list of planets does not change in a given game
    var gameTick: Int=0,
) : RemoteConstructable {
    fun deepCopy(): GameState {
        val copiedPlanets = planets.map { planet ->
            Planet(
                owner = planet.owner,
                nShips = planet.nShips,
                position = Vec2d(planet.position.x, planet.position.y),
                growthRate = planet.growthRate,
                radius = planet.radius,
                transporter = planet.transporter?.let { transporter ->
                    Transporter(
                        s = Vec2d(transporter.s.x, transporter.s.y),
                        v = Vec2d(transporter.v.x, transporter.v.y),
                        owner = transporter.owner,
                        sourceIndex = transporter.sourceIndex,
                        destinationIndex = transporter.destinationIndex,
                        nShips = transporter.nShips,
                    )
                },
                id = planet.id
            )
        }
        return GameState(copiedPlanets, gameTick)
    }
}


fun main() {
    // Create an initial GameState with one planet and a transporter
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
            destinationIndex = 0,
            nShips = 10.0
        ),
        id = 1
    )

    val gameState = GameState(planets = listOf(planet), gameTick = 0)

    // Make a deep copy of the game state
    val copiedState = gameState.deepCopy()

    // Modify the original state to check if the copy remains unaffected
    gameState.planets[0].nShips = 100.0
//    gameState.planets[0].position = Vec2d(200.0, 300.0)
    gameState.planets[0].owner = Player.Player2
//    gameState.planets[0].transporter?.nShips = 20
//    gameState.planets[0].pending[Player.Player1] = 99.0
    gameState.gameTick = 10

    // Output both the original and the copied state to compare
    println("Original GameState after modifications:")
    println(gameState)

    println("\nCopied GameState (should remain unchanged):")
    println(copiedState)
}
