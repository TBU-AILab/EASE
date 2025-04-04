from .llmconnector import LLMConnector
from ..loader import Parameter, PrimitiveType
from ..message import Message
from openai import OpenAI
from ..utils.connector_utils import get_available_models


class LLMConnectorOpenAIImage(LLMConnector):
    """
    LLM Connector for OpenAI Image models.
    :param token: Token ID for the OpenAI
    :param model: Model, set from OpenAIImageModels.
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
        if self._token:
            self._client = OpenAI(api_key=self._token)

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:

        av_models = get_available_models(cls.get_short_name())

        return {
            'token': Parameter(short_name="token", type=PrimitiveType.str),
            'model': Parameter(short_name="model", type=PrimitiveType.enum, long_name='LLM model', enum_options=av_models['model_names'], enum_descriptions=av_models['model_longnames'], default='dall-e-2')
        }

    def _init_params(self):
        super()._init_params()
        self._token = self.parameters.get('token', '')  # Access token, ID
        self._model = self.parameters.get('model', self.get_parameters().get('model').default)

        self._type = 'OpenAIImage'
        if self._token:
            self._client = OpenAI(api_key=self._token)

    ####################################################################
    #########  Public functions
    ####################################################################
    def send(self, context: list[Message]) -> Message:
        _response = self._client.images.generate(
            model=self.get_model(),
            prompt=self._extract_messages(context),
            size='1024x1024',
            quality='standard',
            response_format='b64_json',
            n=1,
        )

        _txt = _response.data[0].revised_prompt
        if _response.data[0].revised_prompt is None:
            _txt = ""
        msg = Message(role=self.get_role_assistant(), model_encoding=None,
                      message=_txt
                      )
        msg.set_metadata(label='image', data=_response.data[0].b64_json)

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
        return "llm.openaiimage"

    @classmethod
    def get_long_name(cls) -> str:
        return "OpenAI Image"

    @classmethod
    def get_description(cls) -> str:
        return "Open AI Image connector. Supports outputs: image."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': set(),
            'output': {'image'}
        }

    ####################################################################
    #########  Private functions
    ####################################################################
    def _extract_messages(self, context: list[Message]) -> str:
        messages = ''
        for cnx in context:
            messages += cnx.get_content()
            messages += '\n'

        return messages
