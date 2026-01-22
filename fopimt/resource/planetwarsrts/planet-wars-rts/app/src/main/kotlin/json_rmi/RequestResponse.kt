package json_rmi

import kotlinx.serialization.*
import kotlinx.serialization.json.*

object RpcConstants {
    const val TYPE_INIT = "init"
    const val TYPE_INVOKE = "invoke"
    const val TYPE_END = "end"
    const val TARGET_AGENT = "agent"
}

interface RemoteConstructable

@Serializable
data class RemoteInvocationRequest(
    val requestType: String,
    val target: String,
    val className: String? = null,
    val method: String,
    val objectId: String? = null,
    val args: List<JsonElement> = emptyList()
) : RemoteConstructable

@Serializable
data class RemoteInvocationResponse(
    val status: String,
    val result: JsonElement? = null,
    val error: String? = null
) : RemoteConstructable

//val json = Json { prettyPrint = true; encodeDefaults = true }
