from ..solutions.solution import Solution
from .test import Test, TestResult


class TestSyntaxPython(Test):
    """
    Test designed for correct Python syntax.
    """

    def _init_params(self):
        super()._init_params()
        self._error_msg = "Test:PythonSyntax: OK"
        self._user_msg = "Test:PythonSyntax: OK"
        self._error_msg_template = "I got a syntax error in your generated code. The error message was: {0}. Fix the error."
        self._user_msg_template = "There seems to be a Syntax Error in the generated Python code. The error message was {0}."

    ####################################################################
    #########  Public functions
    ####################################################################
    def test(self, solution: Solution) -> TestResult:
        """
        This function tests whether the solution file is a python script without syntax errors
        :param solution: Solution to test
        :return: Returns test result (True = test was OK, False = test was not OK)
        """
        self._result = True
        try:
            compile(solution.get_input(), "temp.py", "exec")
        except Exception as e:
            self._result = False
            self._error_msg = self._error_msg_template.format(repr(e))
            self._user_msg = self._user_msg_template.format(repr(e))

        return TestResult(
            class_ref=type(self),
            passed=self._result,
            metadata={
                "error_msg": self._error_msg,
                "user_msg": self._user_msg,
            },
        )

    @classmethod
    def get_short_name(cls) -> str:
        return "test.psyntax"

    @classmethod
    def get_long_name(cls) -> str:
        return "Python syntax test"

    @classmethod
    def get_description(cls) -> str:
        return "Testing of the correct Python syntax."

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": {"python"}, "output": set()}

    ####################################################################
    #########  Private functions
    ####################################################################
