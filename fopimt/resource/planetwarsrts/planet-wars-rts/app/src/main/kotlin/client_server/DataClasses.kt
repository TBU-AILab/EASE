package client_server

import io.ktor.http.*
import io.ktor.websocket.*
import kotlinx.coroutines.*
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import java.time.Instant

@Serializable
data class GameState(val tick: Int, val timestamp: Long)

@Serializable
data class AgentResponse(val tick: Int, val timestamp: Long, val agentId: String, val action: AgentAction = AgentAction())

@Serializable
data class AgentAction(val action: String = "DoNothing")