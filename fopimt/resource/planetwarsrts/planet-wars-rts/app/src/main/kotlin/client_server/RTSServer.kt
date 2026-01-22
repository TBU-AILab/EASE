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
import kotlin.time.Duration.Companion.milliseconds


fun main() {

    val delayMillis = 1000L
    val server = embeddedServer(Netty, port = 8080) {
        install(WebSockets)
        install(ContentNegotiation) {
            json()
        }
        routing {
            webSocket("/game") {
                var tick = 0
                while (true) {
                    val gameState = GameState(tick, Instant.now().toEpochMilli())
                    outgoing.send(Frame.Text(Json.encodeToString(gameState)))
                    withTimeoutOrNull(delayMillis.milliseconds.inWholeMilliseconds) {
                        val frame = incoming.receive()
                        if (frame is Frame.Text) {
                            val response = Json.decodeFromString<AgentResponse>(frame.readText())
                            println("Received response from Agent: $response")
                        }
                    }
                    tick++
                    delay(delayMillis) // simulate game tick delayMillis
                }
            }
        }
    }
    server.start(wait = true)
}
