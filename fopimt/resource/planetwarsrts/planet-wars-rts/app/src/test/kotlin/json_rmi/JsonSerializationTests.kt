package json_rmi

import decodeArgument
import games.planetwars.agents.Action
import org.junit.jupiter.api.Assertions.assertTrue
import games.planetwars.core.*
import kotlinx.serialization.PolymorphicSerializer
import kotlinx.serialization.json.*
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class JsonSerializationTests {

    private val testJson = Json {
        serializersModule = customSerializersModule
        classDiscriminator = "type"
        prettyPrint = true
        encodeDefaults = true
    }

    @Test
    fun `test GameState serialization round-trip`() {
        val original = GameStateFactory(GameParams(numPlanets = 5)).createGame()
        val element: JsonElement =
            testJson.encodeToJsonElement(PolymorphicSerializer(RemoteConstructable::class), original)
        val decoded =
            testJson.decodeFromJsonElement(PolymorphicSerializer(RemoteConstructable::class), element) as GameState

        assertEquals(original.planets.size, decoded.planets.size)
        assertEquals(original.gameTick, decoded.gameTick)
    }

    @Test
    fun `test Player serialization`() {
        val player = Player.Player2
        val element = testJson.encodeToJsonElement(Player.serializer(), player)
        val decoded = testJson.decodeFromJsonElement(Player.serializer(), element)

        assertEquals(player, decoded)
    }

    @Test
    fun `test GameParams serialization`() {
        val params = GameParams(numPlanets = 42)
        val element = testJson.encodeToJsonElement(PolymorphicSerializer(RemoteConstructable::class), params)
        val decoded =
            testJson.decodeFromJsonElement(PolymorphicSerializer(RemoteConstructable::class), element) as GameParams

        assertEquals(params.numPlanets, decoded.numPlanets)
    }

    @Test
    fun `test Action serialization`() {
        val action = Action(Player.Player1, 0, 1, 10.0)
        val element = testJson.encodeToJsonElement(PolymorphicSerializer(RemoteConstructable::class), action)
        val decoded =
            testJson.decodeFromJsonElement(PolymorphicSerializer(RemoteConstructable::class), element) as Action

        assertTrue(action == decoded)
    }

    @Test
    fun `test RemoteInvocationRequest serialization`() {
        val request = RemoteInvocationRequest(
            requestType = RpcConstants.TYPE_INVOKE,
            target = RpcConstants.TARGET_AGENT,
            method = "getAction",
            objectId = "agent-123",
            args = listOf(
                testJson.encodeToJsonElement(PolymorphicSerializer(RemoteConstructable::class), GameParams(10))
            )
        )
        val jsonStr = testJson.encodeToString(RemoteInvocationRequest.serializer(), request)
        val parsed = testJson.decodeFromString(RemoteInvocationRequest.serializer(), jsonStr)
        assertEquals(request.requestType, parsed.requestType)
        assertEquals(request.method, parsed.method)
        assertEquals(request.objectId, parsed.objectId)
        assertEquals(request.target, parsed.target)
    }

    @Test
    fun `test RemoteInvocationResponse serialization`() {
        val response = RemoteInvocationResponse(
            status = "ok",
            result = testJson.encodeToJsonElement(PolymorphicSerializer(RemoteConstructable::class), Action(Player.Player1, 0, 1, 10.0))
        )
        val jsonStr = testJson.encodeToString(RemoteInvocationResponse.serializer(), response)
        val parsed = testJson.decodeFromString(RemoteInvocationResponse.serializer(), jsonStr)
        assertEquals(response.status, parsed.status)
        assertEquals(response.result, parsed.result)
        assertEquals(response.error, parsed.error)
    }

    @Test
    fun `test RemoteInvocationRequest serialization with Polymorphic serializer`() {
        val request = RemoteInvocationRequest(
            requestType = RpcConstants.TYPE_INVOKE,
            target = RpcConstants.TARGET_AGENT,
            method = "getAction",
            objectId = "agent-123",
            args = listOf(
                testJson.encodeToJsonElement(PolymorphicSerializer(RemoteConstructable::class), GameParams(10))
            )
        )

        val jsonStr = testJson.encodeToString(PolymorphicSerializer(RemoteConstructable::class), request)

        println(jsonStr)

        val parsed = testJson.decodeFromString(PolymorphicSerializer(RemoteConstructable::class), jsonStr) as RemoteInvocationRequest
        assertEquals(request.requestType, parsed.requestType)
        assertEquals(request.method, parsed.method)
        assertEquals(request.objectId, parsed.objectId)
        assertEquals(request.target, parsed.target)
    }

    @Test
    fun `test RemoteInvocationRequest with Player`() {
        val request = RemoteInvocationRequest(
            requestType = RpcConstants.TYPE_INVOKE,
            target = RpcConstants.TARGET_AGENT,
            method = "prepareToPlayAs",
            objectId = "some-agent",
            args = listOf(
                testJson.encodeToJsonElement(Player.serializer(), Player.Player1),
                testJson.encodeToJsonElement(PolymorphicSerializer(RemoteConstructable::class), GameParams(10)),
                JsonNull // opponent
            )
        )
        val jsonStr = testJson.encodeToString(RemoteInvocationRequest.serializer(), request)
        val parsed = testJson.decodeFromString(RemoteInvocationRequest.serializer(), jsonStr)
        assertEquals("prepareToPlayAs", parsed.method)
        assertEquals(Player.Player1.name, parsed.args[0].jsonPrimitive.content)
    }

    @Test
    fun `test decodeArgument on Player and GameParams`() {
        val playerJson = testJson.encodeToJsonElement(Player.serializer(), Player.Player2)

        val paramsJson = testJson.encodeToJsonElement(
            PolymorphicSerializer(RemoteConstructable::class),
            GameParams(numPlanets = 7)
        )

        val playerParam = ::dummyFunction.parameters[1]  // Player
        val paramParam = ::dummyFunction.parameters[2]  // GameParams

        val decodedPlayer = decodeArgument(playerParam, playerJson)
        val decodedParams = decodeArgument(paramParam, paramsJson)

        assertEquals(Player.Player2, decodedPlayer)
        assertEquals(7, (decodedParams as GameParams).numPlanets)
    }

    @Suppress("UNUSED_PARAMETER")
    fun dummyFunction(p1: String, p2: Player, p3: GameParams) {}

}
