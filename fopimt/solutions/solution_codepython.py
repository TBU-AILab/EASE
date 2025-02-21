import os
from sys import prefix

from ..message import Message
from .solution import Solution


class SolutionCodePython(Solution):
    """
    Solution Python code type class.
    Gets input (message.context) from Message
    :param prefix: String prefix for the file.
    """
    def _init_params(self):
        super()._init_params()
        self._suffix = '.py'
        self._prefix = self.parameters.get('prefix', '')

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_input_from_msg(self, msg: Message):
        plain = msg.get_content()
        state = 'TEXT'
        out = ""
        for line in plain.split('\n'):
            if line.startswith('```'):
                if state == 'TEXT':
                    state = 'CODE'
                else:
                    state = 'TEXT'
                continue
            if state == 'CODE':
                out += line
                out += '\n'
            if state == 'TEXT':
                out += '#'
                out += line
                out += '\n'

        self._input = out

    @classmethod
    def get_short_name(cls) -> str:
        return "sol.codepython"

    @classmethod
    def get_long_name(cls) -> str:
        return "Python code"

    @classmethod
    def get_description(cls) -> str:
        return "Solution that provide Python code."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'text'},
            'output': {'python'}
        }

    ####################################################################
    #########  Private functions
    ####################################################################
