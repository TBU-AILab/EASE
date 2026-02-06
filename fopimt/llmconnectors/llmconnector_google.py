from .llmconnector import LLMConnector
from ..loader import Parameter, PrimitiveType
from ..message import Message
from google import genai
from google.genai import types
from ..utils.connector_utils import get_available_models
import json


class LLMConnectorGoogle(LLMConnector):

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the client from the state to allow pickling
        if '_client' in state:
            del state['_client']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the client after unpickling
        if self._token:
            self._client = genai.Client(api_key=self._token)

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:

        av_models = get_available_models(cls.get_short_name())

        return {
            'token': Parameter(short_name="token", type=PrimitiveType.str),
            'model': Parameter(short_name="model", type=PrimitiveType.enum, long_name='LLM model', enum_options=av_models['model_names'], enum_descriptions=av_models['model_longnames'], default='gpt-4o-mini')
        }

    def _init_params(self):
        super()._init_params()
        self._token = self.parameters.get('token', '')  # Access token, ID
        self._model = self.parameters.get('model', self.get_parameters().get('model').default)
        self._system_msg = ''

        self._type = 'Google'
        if self._token:
            self._client = genai.Client(api_key=self._token)

    ####################################################################
    #########  Public functions
    ####################################################################
    def send(self, context: list[Message]) -> Message:
        msgs = self._extract_messages(context)
        completion = self._client.models.generate_content(
            model=self._model,
            config=types.GenerateContentConfig(
            system_instruction=self._system_msg),
            contents=msgs
        )

        msg = Message(role=self.get_role_assistant(), model_encoding=None,
                      message=completion.text
                      )
        msg.set_tokens(0)
        if completion.usage_metadata is not None and completion.usage_metadata.total_token_count is not None:
            msg.set_tokens(completion.usage_metadata.total_token_count)

        return msg

    def get_role_user(self) -> str:
        return 'user'

    def get_role_assistant(self) -> str:
        return 'model'

    def get_role_system(self) -> str:
        return 'system'


    def get_model(self) -> str:
        return self._model

    @classmethod
    def get_short_name(cls) -> str:
        return "llm.google"

    @classmethod
    def get_long_name(cls) -> str:
        return "Google"

    @classmethod
    def get_description(cls) -> str:
        return "Google AI connector. Supports outputs: text or code."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': set(),
            'output': {'text'}
        }

        ####################################################################
    #########  Private functions
    ####################################################################
    def _extract_messages(self, context: list[Message]) -> str:
        messages = []
        for cnx in context:
            msg = cnx.get()
            if msg['role'] == self.get_role_system():
                self._system_msg = msg['content']

            messages.append(json.dumps(cnx.get()))

        return str(messages)
