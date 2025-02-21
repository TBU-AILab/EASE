from .llmconnector import LLMConnector
from ..loader import Parameter, PrimitiveType
from ..message import Message
from openai import OpenAI
from enum import Enum
import os


class LLMConnectorOpenAIvision(LLMConnector):

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the client from the state to allow pickling
        if '_client' in state:
            del state['_client']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the client after unpickling
        self._client = OpenAI()

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            'token': Parameter(short_name="token", type=PrimitiveType.str),
            'model': Parameter(short_name="model", type=PrimitiveType.enum, long_name='LLM model', enum_options=[
                'gpt-4o',
                'gpt-4o-mini',
                'gpt-4-turbo',
            ], default='gpt-4o-mini')
        }

    def _init_params(self):
        super()._init_params()
        self._token = self.parameters.get('token', '')  # Access token, ID
        self._model = self.parameters.get('model', self.get_parameters().get('model').default)

        self._type = 'OpenAI'
        if self._token:
            os.environ['OPENAI_API_KEY'] = self._token
        self._client = OpenAI()

    ####################################################################
    #########  Public functions
    ####################################################################
    def send(self, context: list[Message]) -> Message:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=self._extract_messages(context)
        )

        msg = Message(role=self.get_role_assistant(), model_encoding=None,
                      message=completion.choices[0].message.content
                      )
        msg.set_tokens(completion.usage.completion_tokens)

        return msg

    def get_role_user(self) -> str:
        return 'user'

    def get_role_assistant(self) -> str:
        return 'assistant'

    def get_role_system(self) -> str:
        return 'system'

    def get_model(self) -> str:
        return self._model

    @classmethod
    def get_short_name(cls) -> str:
        return "llm.openaivision"

    @classmethod
    def get_long_name(cls) -> str:
        return "OpenAIvision"

    @classmethod
    def get_description(cls) -> str:
        return "Open AI Vision connector. Supports outputs: text."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': set(),
            'output': {'text'}
        }

        ####################################################################
    #########  Private functions
    ####################################################################
    def _extract_messages(self, context: list[Message]):
        messages = []
        for cnx in context:
            if 'image' in cnx.get_metadata():
                _out = {
                    "role": cnx.get_role(),
                    "content": [
                        {"type": "text", "text": cnx.get_content()},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{cnx.get_metadata()['image']}"
                            },
                        },
                    ],
                }
                messages.append(_out)
            else:
                messages.append(cnx.get())

        return messages
