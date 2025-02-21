from .llmconnector import LLMConnector
from ..loader import Parameter, PrimitiveType
from ..message import Message
from enum import Enum
import requests

"""
Requires connection via url to the llama model

Local installation:
1. install Ollama from https://ollama.com/download
2. run Ollama
3. in the ollama command line: ollama pull llama3.1

If run locally (default):
host_url: "http://localhost:11434/api/chat" 

"""

class MetaModels(Enum):
    LLAMA_31_8B = 'llama3.1'
    LLAMA_31_70B = 'llama3.1:70b'


class LLMConnectorMeta(LLMConnector):
    """
    LLM Connector for Meta models.
    :param host_url: URL to the llama server
    :param model: Model, set from Meta models.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            'host_url': Parameter(short_name='host_url', type=PrimitiveType.str, long_name='Host url', default="http://localhost:11434/api/chat"),
            'model': Parameter(short_name="model", type=PrimitiveType.enum, long_name='LLM model', enum_options=[
                'llama3.1',
                'llama3.1:70b'
            ], default='llama3.1')
        }

    def _init_params(self):

        super()._init_params()
        self._model = self.parameters.get('model', self.get_parameters().get('model').default)
        self._type = 'Meta'
        self._url = self.parameters.get('host_url', self.get_parameters().get('host_url').default)

    ####################################################################
    #########  Public functions
    ####################################################################
    def send(self, context: list[Message]) -> Message:

        data = {
            "model": self._model,
            "stream": False
        }

        headers = {
            'Content-Type': 'application/json'
        }

        messages = self._extract_messages(context)
        data["messages"] = messages

        response = requests.post(self._url, headers=headers, json=data)


        msg = Message(role=self.get_role_assistant(), model_encoding=None,
                      message=response.json()['message']['content']
                      )

        msg.set_tokens(response.json()["eval_count"])

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
        return "llm.meta"

    @classmethod
    def get_long_name(cls) -> str:
        return "Meta"

    @classmethod
    def get_description(cls) -> str:
        return "Meta connector. Supports outputs: text or code."

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
            messages.append(cnx.get())

        return messages
