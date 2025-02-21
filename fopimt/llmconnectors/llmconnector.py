from ..message import Message
from enum import Enum
from ..modul import Modul
from ..loader import Parameter, PrimitiveType


class LLMConnector(Modul):
    """
    General LLM Connector class.
    Sets low level definitions and provide layer between app and LLM.
    :param short_name: Name of the LLMConnector type. Short version.
    :param long_name: Name of the LLMConnector type. Long version.
    :param description: Description of the LLMConnector type.
    :param tags: Tags associated with the LLMConnector. USed for compatibility checks.
    :param token: Token|ID used for the LLM connector API.
    :param model: Specified model of the LLM.
    """

    def _init_params(self):
        super()._init_params()
        self._type: str | None = None  # Type of LLM (OpenAI, Meta, Google, ...)

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_role_user(self) -> str:
        """
        Get role specification string for USER.
        Returns string.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_role_system(self) -> str:
        """
        Get role specification string for SYSTEM.
        Returns string.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_role_assistant(self) -> str:
        """
        Get role specification string for ASSISTANT.
        Returns string.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def send(self, context) -> Message:
        """
        Send context to LLM.
        Returns response as Message from LLM.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_model(self) -> str:
        """
        Returns current LLM model as string.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    ####################################################################
    #########  Private functions
    ####################################################################
