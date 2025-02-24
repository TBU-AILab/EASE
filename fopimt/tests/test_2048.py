from .test import Test
from ..solutions.solution import Solution
import numpy as np
from ..utils.import_utils import dynamic_import
import logging
from ..resource.game2048.implementation_2048 import play_2048

from ..loader import Parameter, PrimitiveType

import time
from ..resource.metahuristic.metaheuristic_runner import Runner

MAX_EVAL_TIME = 5

def test_the_game(move) -> dict:
    returndict = {}

    grid, score = np.array([[8,4,2,0],[256,128,64,0],[8,4,2,0],[256,128,64,0]]), 0

    start = time.time()
    direction = move(grid, score)
    end = time.time()

    run_time = end - start

    returndict['pass'] = True
    returndict['reason'] = []

    if direction != 'right':
        returndict['pass'] = False
        returndict['reason'].append(f'Invalid move: {direction}')

    if run_time > MAX_EVAL_TIME:
        returndict['pass'] = False
        returndict['reason'].append(f"Too slow: {run_time}s, only {MAX_EVAL_TIME}s allowed for each move.")

    return returndict


class Test2048(Test):
    """
    This class serves for 2048 solver testing.
    Tests common issues with 2048 solvers:
    - Invalid moves
    - Time constraint
    """

    def _init_params(self):
        super()._init_params()
        # Default messages are used only when the test passes
        self._error_msg = "Test:2048: OK"
        self._user_msg = "Test:2048: OK"

    @classmethod
    def get_short_name(cls) -> str:
        return "test.2048"


    @classmethod
    def get_long_name(cls) -> str:
        return "Solver 2048 test"

    @classmethod
    def get_description(cls) -> str:
        return "This class serves for 2048 solver testing. Tests common issues with 2048 solvers:\n" \
                                     "- Invalid moves\n" \
                                     "- High time requirements\n"

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
            self._error_msg = self._user_msg = f"Test:2048: Algorithm could not be checked due to the following error: {repr(e)}"
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
                        logging.error("Test:2048:", repr(e))

            move = combined_scope['move']
            result = test_the_game(move)

            if result['pass']:
                self._test_result = True
            else:
                err_msg = "Test:2048: Solver encountered the following errors:\n"
                for r in result['reason']:
                    err_msg += (r + "\n")

                self._error_msg = self._user_msg = err_msg
                self._test_result = False

        except Exception as e:
            self._test_result = False
            self._error_msg = self._user_msg = (f"Test:2048: Solver could not be checked due to the "
                                                f"following error: {repr(e)}")
        return self._test_result

