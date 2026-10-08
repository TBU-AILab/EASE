import time

from openai import OpenAI

from ..loader_dto import Parameter, PrimitiveType
from ..message import Message
from ..utils.connector_utils import get_available_models
from .llmconnector import LLMConnector, LLMConnectorResult


class LLMConnectorCerit(LLMConnector):
    """
    Connector for the CERIT-SC AI-as-a-Service API (e-INFRA CZ), https://docs.cerit.io/en/docs/ai-as-a-service/ai-api.
    The service is OpenAI compatible (chat completions), so this is the OpenAI connector with a different base URL
    and models. The API key is generated in the Open WebUI at https://chat.ai.e-infra.cz (Settings > Account > API keys).
    :param token: API key.
    :param model: Model ID (list: GET https://llm.ai.e-infra.cz/v1/models).
    :param base_url: Base URL of the API.
    :param timeout: Request timeout in seconds (the service limits non-streaming requests to 30 minutes).
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
        if self._token:
            self._client = self._create_client()

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
                default=av_models["model_names"][0]
                if av_models["model_names"]
                else None,
            ),
            "base_url": Parameter(
                short_name="base_url",
                type=PrimitiveType.str,
                long_name="Base URL",
                description="Base URL of the OpenAI compatible API.",
                default="https://llm.ai.e-infra.cz/v1",
            ),
            "timeout": Parameter(
                short_name="timeout",
                type=PrimitiveType.int,
                long_name="Timeout [s]",
                description="Request timeout in seconds (the service limits non-streaming requests to 30 minutes).",
                default=1800,
            ),
        }

    def _init_params(self):
        super()._init_params()
        defaults = self.get_parameters()
        self._token = self.parameters.get("token", "")  # Access token, ID
        self._model = self.parameters.get("model", defaults["model"].default)
        self._base_url = self.parameters.get("base_url", defaults["base_url"].default)
        self._timeout = int(self.parameters.get("timeout", defaults["timeout"].default))

        self._type = "CERIT"
        if self._token:
            self._client = self._create_client()

    ####################################################################
    #########  Public functions
    ####################################################################
    def send(self, context: list[Message]) -> LLMConnectorResult:
        t_start = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self._model, messages=self._extract_messages(context)
        )
        duration_s = time.perf_counter() - t_start

        msg = Message(
            role=self.get_role_assistant(),
            model_encoding=None,
            message=completion.choices[0].message.content,
        )

        usage = completion.usage
        if usage is not None:
            msg.set_tokens(usage.completion_tokens)
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            completion_details = getattr(usage, "completion_tokens_details", None)
            self._set_usage(
                msg,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cached_input_tokens=getattr(prompt_details, "cached_tokens", 0),
                cache_write_input_tokens=getattr(
                    prompt_details, "created_cache_tokens", 0
                ),
                # note: the service reports reasoning_tokens = 0 even for reasoning models,
                # the reasoning is included in completion_tokens
                reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0),
                duration_s=duration_s,
            )
        # without usage the Task ledger falls back to the common tokenizer (marked as estimated)

        return LLMConnectorResult(
            class_ref=type(self),
            response=msg,
        )

    def get_role_user(self) -> str:
        return "user"

    def get_role_assistant(self) -> str:
        return "assistant"

    def get_role_system(self) -> str:
        return "system"

    def get_model(self) -> str:
        return self._model

    @classmethod
    def get_short_name(cls) -> str:
        return "llm.cerit"

    @classmethod
    def get_long_name(cls) -> str:
        return "CERIT-SC AI (e-INFRA CZ)"

    @classmethod
    def get_description(cls) -> str:
        return (
            "CERIT-SC AI-as-a-Service connector (OpenAI compatible API of e-INFRA CZ, https://llm.ai.e-infra.cz). "
            "Supports outputs: text or code."
        )

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": set(), "output": {"text"}}

    ####################################################################
    #########  Private functions
    ####################################################################
    def _create_client(self) -> OpenAI:
        return OpenAI(
            api_key=self._token, base_url=self._base_url, timeout=self._timeout
        )

    def _extract_messages(self, context: list[Message]):
        return [cnx.get() for cnx in context]
