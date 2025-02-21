from .test import Test
from ..solutions.solution import Solution
import numpy as np
from ..utils.import_utils import dynamic_import
import logging

from ..loader import Parameter, PrimitiveType

from ..resource.metahuristic.metaheuristic_runner import Runner


class TestMetaheuristic(Test):
    """
    This class serves for metaheuristic testing.
    Tests common issues with metaheuristic algorithms:
    - Maximum evaluation count exceeded
    - Search space bounds breaches
    - Evaluation of wrong length individuals
    :param max_evals: Maximum number of evaluations
    """

    def _init_params(self):
        super()._init_params()
        # Default messages are used only when the test passes
        self._error_msg = "Test:Metaheuristic: OK"
        self._user_msg = "Test:Metaheuristic: OK"
        self._max_evals = 997   # Default value maybe changed in the future as optional argument

    @classmethod
    def get_short_name(cls) -> str:
        return "test.meta"


    @classmethod
    def get_long_name(cls) -> str:
        return "Metaheuristic"

    @classmethod
    def get_description(cls) -> str:
        return "This class serves for metaheuristic testing. Tests common issues with metaheuristic algorithms:\n" \
                                     "- Maximum evaluation count exceeded\n" \
                                     "- Search space bounds breaches\n" \
                                     "- Evaluation of wrong length individuals\n"

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'python'},
            'output': set()
        }

    def test(self, solution: Solution) -> bool:
        """
        This method tests the given solution for bound crossing

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
            self._error_msg = self._user_msg = f"Test:Metaheuristic: Algorithm could not be checked due to the following error: {repr(e)}"
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
                        logging.error("Test:Poster:", repr(e))

            alg = combined_scope['run']
            dim = 10
            func = TestMetaheuristic.InvertedSphere(dim)
            a = Runner(alg, func, dim, func.get_bounds(), self._max_evals)
            result = a.run()
        except Exception as e:
            self._test_result = False
            self._error_msg = self._user_msg = (f"Test:Metaheuristic: Algorithm could not be checked due to the "
                                                f"following error: {repr(e)}")
            return self._test_result

        # Check for any errors during evaluation
        # Maximum evaluations exceeded - not severe
        if result.get('maxevalexception', False):
            self._error_msg = self._user_msg = ('Test:Metaheuristic: Algorithm tried to exceed maximum number of '
                                                'evaluations.')
            self._test_result = True
        # Out of bounds - not severe
        if result.get('outofboundsexception', False):
            self._error_msg = self._user_msg = ('Test:Metaheuristic: Algorithm is crossing the bounds of the specified '
                                                'search space, check the bounds properly.')
            self._test_result = True
        # Wrong dimensionality
        if result.get('dimexception', False):
            self._error_msg = self._user_msg = ('Test:Metaheuristic: Algorithm tried to evaluate solution of '
                                                'different dimension than the problem specification.')
            self._test_result = False
        # Unspecified exception
        if result.get('unexpectedexception', False):
            self._error_msg = self._user_msg = f'Test:Metaheuristic: {result.get("unexpectedexception")}'
            self._test_result = False

        return self._test_result

    class InvertedSphere:

        def __init__(self, dim):
            """
            Testing function
            """
            self._dim = dim

        def __str__(self) -> str:
            return 'InvertedSphere'

        def get_bounds(self):
            return np.array([[-1, 1]] * self._dim)

        def evaluate(self, x: np.array) -> float:
            if type(x) is not np.array:
                x = np.array(x)

            return -np.sum(x ** 2)
