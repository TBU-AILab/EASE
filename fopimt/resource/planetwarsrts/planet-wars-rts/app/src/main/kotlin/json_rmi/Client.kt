package json_rmi

import games.planetwars.core.Player
import io.ktor.client.plugins.websocket.DefaultClientWebSocketSession
import io.ktor.websocket.*
import kotlinx.serialization.PolymorphicSerializer
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.*

val json = Json {
    serializersModule = customSerializersModule
    classDiscriminator = "type"
    prettyPrint = true
    encodeDefaults = true
    ignoreUnknownKeys = true
}

fun encodeArgument(arg: Any?): JsonElement = when (arg) {
    is RemoteConstructable -> json.encodeToJsonElement(PolymorphicSerializer(RemoteConstructable::class), arg)
    is Player -> json.encodeToJsonElement(Player.serializer(), arg)
    is String -> JsonPrimitive(arg)
    is Int -> JsonPrimitive(arg)
    is Double -> JsonPrimitive(arg)
    is Boolean -> JsonPrimitive(arg)
    else -> error("Unsupported argument type: ${arg?.let { it::class.simpleName } ?: "null"}")
}

suspend fun invokeRemoteMethod(
    session: DefaultClientWebSocketSession,
    objectId: String,
    method: String,
    args: List<Any?> = emptyList(),
    logger: JsonLogger = JsonLogger(ignore = true),
): String {
    val jsonArgs: List<JsonElement> = args.map(::encodeArgument)
    val request = RemoteInvocationRequest(
        requestType = RpcConstants.TYPE_INVOKE,
        target = RpcConstants.TARGET_AGENT,
        method = method,
        objectId = objectId,
        args = jsonArgs
    )
    val jsonRequest = json.encodeToString(RemoteInvocationRequest.serializer(), request)
    logger.logSend(jsonRequest)
    session.send(jsonRequest)

    val response = (session.incoming.receive() as Frame.Text).readText()
    logger.logRecv(response)
    return response
}

suspend fun initAgent(
    session: DefaultClientWebSocketSession,
    className: String,
    logger: JsonLogger = JsonLogger(ignore = true),
): String {
    val request = RemoteInvocationRequest(
        requestType = RpcConstants.TYPE_INIT,
        target = RpcConstants.TARGET_AGENT,
        className = className,
        method = "<ignored>"
    )
    val jsonRequest = json.encodeToString(RemoteInvocationRequest.serializer(), request)
    logger.logSend(jsonRequest)
    session.send(jsonRequest)

    val initResponse = (session.incoming.receive() as Frame.Text).readText()
    logger.logRecv(initResponse)

    return json.parseToJsonElement(initResponse)
        .jsonObject["result"]
        ?.jsonObject?.get("objectId")
        ?.jsonPrimitive?.content
        ?: error("Missing objectId from INIT response")
}

suspend fun endAgent(
    session: DefaultClientWebSocketSession,
    objectId: String,
    logger: JsonLogger = JsonLogger(ignore = true),
) {
    val request = RemoteInvocationRequest(
        requestType = RpcConstants.TYPE_END,
        target = RpcConstants.TARGET_AGENT,
        method = "",
        objectId = objectId
    )
    val jsonRequest = json.encodeToString(RemoteInvocationRequest.serializer(), request)
    logger.logSend(jsonRequest)
    session.send(jsonRequest)

    val response = (session.incoming.receive() as Frame.Text).readText()
    logger.logRecv(response)
}
