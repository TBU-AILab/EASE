import os
import base64
from sys import prefix

from ..message import Message
from .solution import Solution


class SolutionImage(Solution):
    """
    Solution Image type class.
    Gets input (message.context) from Message
    :param prefix: String prefix for the file.
    """

    def _init_params(self):
        super()._init_params()
        self._suffix = '.png'
        self._prefix = self.parameters.get('prefix', '')

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_input_from_msg(self, msg: Message):
        # Images are stored in 'image' metadata in Message
        self._input = msg.get_metadata()['image']

    def export(self, dir: str, id: str) -> None:
        # export solution itself (code, text, ...)
        file_name = self._prefix + id + self._suffix
        self._path = os.path.join(dir, file_name)
        self._metadata['url'] = self._path
        file = open(self._path, 'wb')
        file.write(base64.b64decode(self._input))
        file.close()

    @classmethod
    def get_short_name(cls) -> str:
        return "sol.image"

    @classmethod
    def get_long_name(cls) -> str:
        return "Image PNG"

    @classmethod
    def get_description(cls) -> str:
        return "Solution that provide PNG Image."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'image'},
            'output': {'image'}
        }
    ####################################################################
    #########  Private functions
    ####################################################################
