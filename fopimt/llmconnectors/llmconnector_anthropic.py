import anthropic

from ..loader_dto import Parameter, PrimitiveType
from ..message import Message
from ..utils.connector_utils import get_available_models
from .llmconnector import LLMConnector, LLMConnectorResult


class LLMConnectorAnthropic(LLMConnector):
    """
    LLM Connector class for Anthropic models.
    :param token: Token|ID used for the LLM connector API.
    :param model: Specified model of the LLM.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the client from the state to allow pickling
        if "_client" in state:
            del state["_client"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the client after unpickling
        self._client = anthropic.Anthropic(api_key=self._token)

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        av_models = get_available_models(cls.get_short_name())

        return {
            "token": Parameter(
                short_name="token", type=PrimitiveType.str, sensitive=True
            ),
            "model": Parameter(
                short_name="model",
                type=PrimitiveType.enum,
                long_name="LLM model",
                enum_options=av_models["model_names"],
                enum_descriptions=av_models["model_longnames"],
                default="claude-3-haiku-20240307",
            ),
        }

    def _init_params(self):
        super()._init_params()
        self._token = self.parameters.get("token", "")  # Access token, ID
        self._model = self.parameters.get(
            "model", self.get_parameters().get("model").default
        )

        self._type = "Anthropic"  # Type of LLM (OpenAI, Meta, Google, ...)
        self._system_msg = ""
        self._client = anthropic.Anthropic(api_key=self._token)

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_role_user(self) -> str:
        """
        Get role specification string for USER.
        Returns string.
        """
        return "user"

    def get_role_system(self) -> str:
        """
        Get role specification string for SYSTEM.
        Returns string.
        """
        return "system"

    def get_role_assistant(self) -> str:
        """
        Get role specification string for ASSISTANT.
        Returns string.
        """
        return "assistant"

    def send(self, context) -> LLMConnectorResult:
        msgs = self._extract_messages(context)

        HARD_MAX_TOKENS = 65536  # 64k per-call output
        MAX_CONTINUATIONS = 20  # safety cap

        all_text_parts = []
        total_output_tokens = 0

        working_msgs = list(msgs)

        for _ in range(MAX_CONTINUATIONS + 1):
            chunk_text = ""

            # Streaming required for long requests (Anthropic SDK constraint)
            with self._client.messages.stream(
                model=self._model,
                system=self._system_msg,
                messages=working_msgs,
                max_tokens=HARD_MAX_TOKENS,
            ) as stream:
                for delta_text in stream.text_stream:
                    chunk_text += delta_text
                final_msg = stream.get_final_message()

            all_text_parts.append(chunk_text)

            # Usage token accounting (best-effort)
            try:
                usage = getattr(final_msg, "usage", None)
                if (
                    usage is not None
                    and getattr(usage, "output_tokens", None) is not None
                ):
                    total_output_tokens += int(usage.output_tokens)
            except Exception:
                pass

            stop_reason = getattr(final_msg, "stop_reason", None)

            # Finished normally (e.g., "end_turn", "stop_sequence", etc.)
            if stop_reason != "max_tokens":
                break

            # Doc-style continuation: append assistant output, then ask to continue without repetition
            working_msgs.append(
                {"role": self.get_role_assistant(), "content": chunk_text}
            )
            working_msgs.append(
                {
                    "role": self.get_role_user(),
                    "content": (
                        "Continue exactly where you left off. "
                        "Do not repeat anything already written. "
                        "Do not add any preamble like 'Continuing'."
                    ),
                }
            )
        # else: exhausted continuations; return what we have

        formatted_output = "".join(all_text_parts)

        msg = Message(
            role=self.get_role_assistant(),
            model_encoding=None,
            message=formatted_output,
        )
        if total_output_tokens:
            msg.set_tokens(total_output_tokens)

        return LLMConnectorResult(
            class_ref=type(self),
            response=msg,
        )

    def get_model(self) -> str:
        """
        Returns current LLM model as string.
        """
        return self._model

    @classmethod
    def get_short_name(cls) -> str:
        return "llm.anthropic"

    @classmethod
    def get_long_name(cls) -> str:
        return "Anthropic LLM connector"

    @classmethod
    def get_description(cls) -> str:
        return "Connector for Anthropic LLM family of models."

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": set(), "output": {"text"}}

    ####################################################################
    #########  Private functions
    ####################################################################
    def _extract_messages(self, context: list[Message]):
        messages = []
        for cnx in context:
            msg = cnx.get()
            if msg["role"] == self.get_role_system():
                self._system_msg = msg["content"]
            else:
                messages.append(cnx.get())

        return messages
