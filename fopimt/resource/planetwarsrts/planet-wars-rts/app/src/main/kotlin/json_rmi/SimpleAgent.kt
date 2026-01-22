package json_rmi


// File: SimpleAgent.kt
interface SimpleAgent {
    fun init(name: String): String
    fun greet(): String
}

class EchoAgent : SimpleAgent {
    private var name: String = ""

    override fun init(name: String): String {
        this.name = name
        return "Initialized with $name"
    }

    override fun greet(): String = "Hello, $name!"
}
