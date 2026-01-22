from .test import Test
from ..solutions.solution import Solution
from ..utils.import_utils import dynamic_import
import logging

class TestTransition(Test):
    """
    This class serves for transition prediction testing.
    Tests common issues with transition prediction solvers:
    - cannot extract features from the .ts files
    """

    def _init_params(self):
        super()._init_params()
        # Default messages are used only when the test passes
        self._error_msg = "Test:Transition: OK"
        self._user_msg = "Test:Transition: OK"

    @classmethod
    def get_short_name(cls) -> str:
        return "test.transition"


    @classmethod
    def get_long_name(cls) -> str:
        return "Transition prediction test"

    @classmethod
    def get_description(cls) -> str:
        return "This class serves for transition prediction testing."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'python'},
            'output': set()
        }

    def test(self, solution: Solution) -> bool:
        """
        This method tests the given solution

        :param solution: Solution to test
        :return: Returns test result (True = test was OK, False = test was not OK)
        """

        self._test_result = True
        local_scope = {}

        # Dynamic import of solution-specific libraries
        exec_globals = {}
        if 'modules' in solution.get_metadata().keys():
            imports = solution.get_metadata()['modules']
            for module_name, specific_part, alias in imports:
                dynamic_import(module_name, specific_part, alias, exec_globals)

        try:
            compile(solution.get_input(), "temp.py", "exec")
            exec(solution.get_input(), exec_globals, local_scope)
        except Exception as e:
            self._test_result = False
            self._error_msg = self._user_msg = f"Test:Transition: Algorithm could not be checked due to the following error: {repr(e)}"
            return self._test_result

        try:

            # Merge exec_globals and exec_locals to ensure functions can access each other
            combined_scope = {**exec_globals, **local_scope}

            # Rebind the global scope for all functions defined in the script
            for key, value in combined_scope.items():
                if callable(value) and not isinstance(value, type):  # If the value is a function
                    try:
                        value.__globals__.update(combined_scope)  # Update its global scope
                    except Exception as e:
                        logging.error("Test:Transition:", repr(e))

            predict = combined_scope['predict']

            X_train = [["/data/ModernTV/Nickelodeon/2024-11-21/1732146475-0023-00864936-00315942.ts"]]
            y_train = [[1]]
            X_test = [["/data/ModernTV/Nickelodeon/2024-11-21/1732146475-0023-00864936-00315942.ts"]]

            y_pred = predict(X_train, y_train, X_test)

        except Exception as e:
            self._test_result = False
            self._error_msg = self._user_msg = (f"Test:Transition: Solution could not be checked due to the "
                                                f"following error: {repr(e)}")
        return self._test_result

