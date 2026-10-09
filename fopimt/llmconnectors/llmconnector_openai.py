import time

from openai import OpenAI

from ..loader_dto import Parameter, PrimitiveType
from ..message import Message
from ..utils.connector_utils import get_available_models
from .llmconnector import LLMConnector, LLMConnectorResult


class LLMConnectorOpenAI(LLMConnector):
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
            self._client = OpenAI(api_key=self._token)

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
        }

    def _init_params(self):
        super()._init_params()
        self._token = self.parameters.get("token", "")  # Access token, ID
        self._model = self.parameters.get(
            "model", self.get_parameters().get("model").default
        )

        self._type = "OpenAI"
        if self._token:
            self._client = OpenAI(api_key=self._token)

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
        msg.set_tokens(completion.usage.completion_tokens)

        usage = completion.usage
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        self._set_usage(
            msg,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cached_input_tokens=getattr(prompt_details, "cached_tokens", 0),
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0),
            duration_s=duration_s,
        )

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
        return "llm.openai"

    @classmethod
    def get_long_name(cls) -> str:
        return "OpenAI"

    @classmethod
    def get_description(cls) -> str:
        return "Open AI connector. Supports outputs: text or code."

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": set(), "output": {"text"}}

        ####################################################################

    #########  Private functions
    ####################################################################
    def _extract_messages(self, context: list[Message]):
        messages = []
        for cnx in context:
            messages.append(cnx.get())

        return messages
