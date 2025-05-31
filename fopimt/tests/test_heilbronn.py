from .test import Test
from ..solutions.solution import Solution
import numpy as np
from ..utils.import_utils import dynamic_import
import logging
import time
import itertools

MAX_EVAL_TIME = 10

class TestHeilbronn(Test):
    """
    This class serves for Heilbronn algorithm testing.
    """

    def _init_params(self):
        super()._init_params()
        # Default messages are used only when the test passes
        self._error_msg = "Test:Heilbronn: OK"
        self._user_msg = "Test:Heilbronn: OK"

    @classmethod
    def get_short_name(cls) -> str:
        return "test.Heilbronn"


    @classmethod
    def get_long_name(cls) -> str:
        return "Heilbronn Triangle test"

    @classmethod
    def get_description(cls) -> str:
        return "This class serves for Heilbronn Triangle solver testing. Tests common issues with solutions:\n" \
                                     "- Unhandled time constraint\n" \
                                     "- Solution outside specified triangle\n"

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
            self._error_msg = self._user_msg = f"Test:Heilbronn: Algorithm could not be checked due to the following error: {repr(e)}"
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
                        logging.error("Test:Heilbronn:", repr(e))

            algorithm = combined_scope['run']

            start_time = time.time()
            coordinates, _ = algorithm(11, _evaluate_found_points, _check_inside_triangle, MAX_EVAL_TIME)
            end_time = time.time()

            if end_time - start_time > (MAX_EVAL_TIME+1):
                self._error_msg = self._user_msg = f'Test:Heilbronn: Generated algorithm violated the time constraint ({MAX_EVAL_TIME} +1 (for orchestration)s), evaluation time was: {end_time - start_time}s'
                self._test_result = False

            if not _check_inside_triangle(coordinates):
                self._error_msg = self._user_msg = 'Test:Heilbronn: Generated algorithm violated the triangle constraint.'
                self._test_result = False

        except Exception as e:
            self._test_result = False
            self._error_msg = self._user_msg = (f"Test:Heilbronn: Algorithm could not be checked due to the "
                                                f"following error: {repr(e)}")
        return self._test_result

def _check_inside_triangle(points: np.ndarray) -> bool:
    """
    Returns True if all points are inside or on the boundary of the equilateral triangle
    with vertices (0,0), (1,0), and (0.5, sqrt(3)/2). Returns False otherwise.
    """
    for (x, y) in points:
        if not (
                (y >= 0) and
                (y <= np.sqrt(3) * x) and
                (y <= -np.sqrt(3) * x + np.sqrt(3))
        ):
            return False
    return True

def _triangle_area(a: np.array, b: np.array, c: np.array) -> float:
    return np.abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2

def _evaluate_found_points(points: np.ndarray):

    a = np.array([0, 0])
    b = np.array([1, 0])
    c = np.array([0.5, np.sqrt(3) / 2])

    if _check_inside_triangle(points):
        min_triangle_area = min(
            [_triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)])
        # Normalize the minimum triangle area (since the equilateral triangle is not unit).
        min_area_normalized = min_triangle_area / _triangle_area(a, b, c)
        return min_area_normalized
    else:
        return -1

