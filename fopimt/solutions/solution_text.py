from ..message import Message
from .solution import Solution


class SolutionText(Solution):
    """
    Solution Text type class.
    Gets input (message.context) from Message
    """

    def _init_params(self):
        super()._init_params()
        self._suffix = '.txt'
        self._prefix = self.parameters.get('prefix', '')

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_input_from_msg(self, msg: Message):
        self._input = msg.get_content()


    @classmethod
    def get_short_name(cls) -> str:
        return "sol.text"

    @classmethod
    def get_long_name(cls) -> str:
        return "Text"

    @classmethod
    def get_description(cls) -> str:
        return "Solution of the text type."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'text'},
            'output': {'text'}
        }

    ####################################################################
    #########  Private functions
    ####################################################################
