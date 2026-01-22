package client_server

import io.ktor.server.application.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.routing.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.response.*
import io.ktor.server.request.*
import io.ktor.server.websocket.*
import io.ktor.websocket.*
import kotlinx.coroutines.*
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import java.time.Instant
import java.util.concurrent.ConcurrentHashMap
import kotlin.time.Duration.Companion.milliseconds


fun main() {
    val delayMillis = 1000L
    val clients = ConcurrentHashMap<WebSocketSession, String>() // Map of connected clients
    val actions = ConcurrentHashMap<String, AgentAction>() // Actions per tick (clientId -> action)
    var tick = 0

    val server = embeddedServer(Netty, port = 8080) {
        install(WebSockets)
        install(ContentNegotiation) {
            json()
        }
        routing {
            webSocket("/game") {
                val clientId = call.parameters["id"] ?: "Client-${this.hashCode()}"
                clients[this] = clientId
                println("Client connected: $clientId")

                try {
                    while (true) {
                        // Send current game state
                        val gameState = GameState(tick, Instant.now().toEpochMilli())
                        outgoing.send(Frame.Text(Json.encodeToString(gameState)))

                        // Receive client response
                        withTimeoutOrNull(delayMillis.milliseconds.inWholeMilliseconds) {
                            val frame = incoming.receive()
                            if (frame is Frame.Text) {
                                val response = Json.decodeFromString<AgentResponse>(frame.readText())
                                actions[clientId] = response.action
                                println("Received response from ${response.agentId} : $clientId: ${response.action}")
                            }
                        }
                        delay(50) // Allow small buffer before proceeding
                    }
                } finally {
                    clients.remove(this)
                    println("Client disconnected: $clientId")
                }
            }
        }

        // Game tick loop
        launch {
            while (true) {
                delay(delayMillis) // Wait for tick duration
                tick++
                println("Processing tick $tick")

                // Process actions and update game state (stubbed)
                for ((clientId, action) in actions) {
                    println("Processing action for $clientId: $action")
                }

                actions.clear() // Reset actions for the next tick

                // Broadcast updated game state to all clients
                val gameState = GameState(tick, Instant.now().toEpochMilli())
                clients.keys.forEach { session ->
                    launch {
                        session.send(Frame.Text(Json.encodeToString(gameState)))
                    }
                }
            }
        }
    }
    server.start(wait = true)
}
