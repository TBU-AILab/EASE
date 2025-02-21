from ..solutions.solution import Solution
from ..modul import Modul
from ..loader import Parameter, PrimitiveType


class Test(Modul):
    """
    General class for definition of a test.
    :param short_name: Name of the Test module. Short version.
    :param long_name: Name of the Test module. Long version.
    :param description: Description of the Test module.
    :param tags: Tags associated with the Test module. Used for the compatibility check.
    """

    def _init_params(self):
        super()._init_params()
        self._test_result: bool = True
        self._error_msg_template: str = ''
        self._user_msg_template: str = ''
        self._error_msg: str = ''
        self._user_msg: str = ''

    @property
    def user_msg_template(self):
        return self._user_msg_template

    @property
    def error_msg_template(self):
        return self._error_msg_template

    ####################################################################
    #########  Public functions
    ####################################################################
    def test(self, solution: Solution) -> bool:
        """
        This method tests the given solution for errors
        Needs to be overwritten in child classes
        :param solution: Solution to test
        :return: Returns test result (True = test was OK, False = test was not OK)
        """

        raise NotImplementedError("The function to test solution has to be implemented")

    def get_test_result(self) -> bool:
        """
        Returns the status of the testing.
        :return: True if the test result is OK, False otherwise
        """
        return self._test_result

    def get_error_msg(self) -> str:
        """
        Returns message (str) defined for LLMConnector feedback.
        :return: Error message with details
        """
        return self._error_msg

    def set_error_msg_template(self, msg: str) -> None:
        """
        Sets error message template used as feedback to LLM.
        It is possible to use formatted string if you want to include the message of the captured error.
        Use as: {0} in string. E.g. "The error was: {0}. Fix it you lazy bastard."
        :param msg: Message text
        :return:
        """
        self._error_msg_template = msg

    def set_user_msg_template(self, msg: str) -> None:
        """
        Sets error message template used as feedback to the user.
        It is possible to use formatted string if you want to include the message of the captured error.
        Use as: {0} in string. E.g. "The error was: {0}. The LLM should fix it."
        :param msg: Message text
        :return:
        """
        self._user_msg_template = msg

    def get_user_msg(self) -> str:
        """
        Returns the user defined message.
        :return: User defined message (str).
        """
        return self._user_msg

    ####################################################################
    #########  Private functions
    ####################################################################
