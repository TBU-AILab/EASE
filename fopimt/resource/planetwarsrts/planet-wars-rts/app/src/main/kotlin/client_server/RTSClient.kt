package client_server

import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.websocket.*
import io.ktor.http.*
import io.ktor.websocket.*
import kotlinx.coroutines.*
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import java.time.Instant

fun main(args: Array<String>) {
    // try to get  agentId from command line argument or else use a default value
    val agentId = if (args.isNotEmpty()) args[0] else "Agnes"
    val client = HttpClient(CIO) {
        install(WebSockets)
    }

    val clientDelay = 1000L

    runBlocking {
        client.webSocket(
            method = HttpMethod.Get,
            host = "127.0.0.1",
            port = 8080,
            path = "/game"
        ) {
            while (true) {
                val frame = incoming.receive()
                if (frame is Frame.Text) {
                    // Decode incoming message
                    val gameState = Json.decodeFromString<GameState>(frame.readText())
                    println("Received GameState: $gameState")
                    // calculate the delayMillis from the timestamp until now
                    // wait for clientDelay before sending the response
                    delay(clientDelay)
                    val delay = Instant.now().toEpochMilli() - gameState.timestamp

                    // Send response back to the server
                    val response = AgentResponse(
                        gameState.tick,
                        delay,
                        agentId
                    )
                    outgoing.send(Frame.Text(Json.encodeToString(response)))
                }
            }
        }
    }
}
