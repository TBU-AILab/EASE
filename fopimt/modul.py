from typing import Any

from .loader import Parameter

class Modul:
    def __init__(self, parameters: dict[str, Any]):
        """
        Basic Modul. All classes in defined packages must inherit it.
        Defines basic description operations for class.
        """
        self.parameters = parameters
        self._init_params()

    ####################################################################
    #########  Public functions
    ####################################################################
    @classmethod
    def get_short_name(cls) -> str:
        raise NotImplementedError("Return short name of the Modul")

    @classmethod
    def get_long_name(cls) -> str:
        raise NotImplementedError("Return long name of the Modul")

    @classmethod
    def get_description(cls) -> str:
        raise NotImplementedError("Return description of the Modul")

    @classmethod
    def get_tags(cls) -> dict:
        raise NotImplementedError("return tags of the Modul")

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        # Maybe overriden by subsequent class
        return dict()

    ####################################################################
    #########  Private functions
    ####################################################################
    def _init_params(self):
        pass