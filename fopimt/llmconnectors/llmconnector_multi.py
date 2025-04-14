from ..message import Message
from enum import Enum
from ..loader import Parameter, PrimitiveType, Loader, PackageType
from .llmconnector import LLMConnector
import random


class LLMConnectorMulti(LLMConnector):
    """
    General LLM Connector class.
    Sets low level definitions and provide layer between app and LLM.
    :param short_name: Name of the LLMConnector type. Short version.
    :param long_name: Name of the LLMConnector type. Long version.
    :param description: Description of the LLMConnector type.
    :param tags: Tags associated with the LLMConnector. Used for compatibility checks.
    :param token: Token|ID used for the LLM connector API.
    :param model: Specified model of the LLM.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        avail_llms = Loader((PackageType.LLMConnector, ), [cls.get_short_name()]).get_package(PackageType.LLMConnector).get_moduls()

        param_dict = {
            'selection_type': Parameter(short_name='selection_type', type=PrimitiveType.enum, long_name='LLM selection',
                                        description='How the LLMs will be selected for output generation.',
                                        enum_options=['random', 'w-random', 'circular'],
                                        enum_descriptions=['Random', 'Weighted random', 'Circular'], default='random',
                                        required=True),
            'llms': Parameter(short_name='llms', long_name='LLM connectors', description='A list of LLM connectors '
                                                                                         'with weights.',
                              type=PrimitiveType.list, required=True,
                              default=[
                                  {
                                      'weight': Parameter(short_name='weight', type=PrimitiveType.float,
                                                          long_name='LLM weight',
                                                          description='Selection weight for this LLM connector. '
                                                                      'Ignored in case of random or circular '
                                                                      'selection type.',
                                                          default=1., required=True),
                                      'llm': Parameter(short_name='llm', type=PrimitiveType.enum,
                                                       long_name='LLM', description='LLM connector',
                                                       enum_options=avail_llms, required=True),
                                  }
                              ]),

        }

        return param_dict

    def _init_params(self):
        super()._init_params()

        self._selection_type = self.parameters.get('selection_type', self.get_parameters().get('selection_type').default)
        self._llms = []
        self._weights = []
        for llm in self.parameters.get('llms', []):
            self._llms.append(Loader().get_package(PackageType.LLMConnector).get_modul_imported(llm['llm']['short_name'])(llm['llm']['parameters']))
            self._weights.append(llm['weight'])

        self._index = 0
        self._llm = None
        self._model = None

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_role_user(self) -> str:
        if self._llm:
            return self._llm.get_role_user()
        return 'user'

    def get_role_assistant(self) -> str:
        if self._llm:
            return self._llm.get_role_assistant()
        return 'assistant'

    def get_role_system(self) -> str:
        if self._llm:
            return self._llm.get_role_system()
        return 'system'

    @classmethod
    def get_short_name(cls) -> str:
        return "llm.multi"

    @classmethod
    def get_long_name(cls) -> str:
        return "Multi-LLM connector"

    @classmethod
    def get_description(cls) -> str:
        return "Multi-LLM connector. Select multiple LLM connectors and type of their selection. Supports outputs: text or code."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': set(),
            'output': {'text'}
        }

    def send(self, context) -> Message:
        """
        Send context to LLM.
        Returns response as Message from LLM.
        """
        match self._selection_type:
            case 'random':
                # Select llm at random
                self._llm = random.choice(self._llms)

            case 'w-random':
                # Weighted random selection
                self._llm = random.choices(self._llms, self._weights)[0]
            case _:
                # Circular
                self._llm = self._llms[self._index]
                self._index += 1
                self._index = self._index % len(self._llms)

        self._model = self._llm.get_model()

        return self._llm.send(context)


    def get_model(self) -> str:
        if self._model:
            return self._model
        else:
            return ""

    ####################################################################
    #########  Private functions
    ####################################################################
