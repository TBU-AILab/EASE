from ..message import Message
from .llmconnector import LLMConnector
from ..loader import Parameter, PrimitiveType
from enum import Enum
import anthropic


class LLMConnectorClaude(LLMConnector):
    """
    LLM Connector class for Claude models.
    :param token: Token|ID used for the LLM connector API.
    :param model: Specified model of the LLM.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the client from the state to allow pickling
        if '_client' in state:
            del state['_client']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the client after unpickling
        self._client = anthropic.Anthropic(api_key=self._token)

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            'token': Parameter(short_name="token", type=PrimitiveType.str),
            'model': Parameter(short_name="model", type=PrimitiveType.enum, long_name='LLM model', enum_options=[
                'claude-3-7-sonnet-20250219',
                'claude-3-5-sonnet-20241022',
                'claude-3-5-haiku-20241022'
                'claude-3-opus-20240229',
                'claude-3-haiku-20240307'
            ], default='claude-3-haiku-20240307')
        }

    def _init_params(self):
        super()._init_params()
        self._token = self.parameters.get('token', '')  # Access token, ID
        self._model = self.parameters.get('model', self.get_parameters().get('model').default)

        self._type = 'Claude'  # Type of LLM (OpenAI, Meta, Google, ...)
        self._system_msg = ''
        self._client = anthropic.Anthropic(api_key=self._token)

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_role_user(self) -> str:
        """
        Get role specification string for USER.
        Returns string.
        """
        return 'user'

    def get_role_system(self) -> str:
        """
        Get role specification string for SYSTEM.
        Returns string.
        """
        return 'system'

    def get_role_assistant(self) -> str:
        """
        Get role specification string for ASSISTANT.
        Returns string.
        """
        return 'assistant'

    def send(self, context) -> Message:

        message = self._client.messages.create(
            max_tokens=4096,
            system=self._system_msg,
            messages=self._extract_messages(context),
            model=self._model,
        )
        response = message.to_dict()
        if 'content' in response:
            if isinstance(response['content'], list):
                formatted_output = "".join(
                    text_block['text'] for text_block in response['content'] if 'text' in text_block)
        else:
            formatted_output = ""

        msg = Message(role=self.get_role_assistant(), model_encoding=None,
                      message=formatted_output
                      )
        msg.set_tokens(message.usage.output_tokens)

        return msg



    def get_model(self) -> str:
        """
        Returns current LLM model as string.
        """
        return self._model

    @classmethod
    def get_short_name(cls) -> str:
        return "llm.claude"

    @classmethod
    def get_long_name(cls) -> str:
        return "Claude LLM connector"

    @classmethod
    def get_description(cls) -> str:
        return "Connector for Claude LLM family of models."

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
            msg = cnx.get()
            if msg['role'] == self.get_role_system():
                self._system_msg = msg['content']
            else:
                messages.append(cnx.get())

        return messages
