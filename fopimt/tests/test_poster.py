from .test import Test
from ..solutions.solution import Solution
from ..utils.import_utils import dynamic_import
from ..resource.resource import Resource, ResourceType
import logging

class TestPoster(Test):
    """
    This class serves for posterization testing.
    """

    def _init_params(self):
        super()._init_params()
        # Default messages are used only when the test passes
        self._error_msg = "Test:Poster: OK"
        self._user_msg = "Test:Poster: OK"


    @classmethod
    def get_short_name(cls) -> str:
        return "test.poster"

    @classmethod
    def get_long_name(cls) -> str:
        return "Poster"

    @classmethod
    def get_description(cls) -> str:
        return "This class serves for poster selection testing. Tests the run of the algorithm on a sample directory with pictures."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'python'},
            'output': set()
        }

    def test(self, solution: Solution) -> bool:
        """

        :param solution: Solution to test
        :return: Returns test result (True = test was OK, False = test was not OK)
        """

        self._test_result = True
        local_scope = {}

        # Dynamic import of solution-specific libraries
        exec_globals = globals()
        if 'modules' in solution.get_metadata().keys():
            imports = solution.get_metadata()['modules']
            for module_name, specific_part, alias in imports:
                dynamic_import(module_name, specific_part, alias, exec_globals)

        try:
            compile(solution.get_input(), "solution_to_evaluate.py", "exec")
            exec(solution.get_input(), exec_globals, local_scope)

            # Merge exec_globals and exec_locals to ensure functions can access each other
            combined_scope = {**exec_globals, **local_scope}

            # Rebind the global scope for all functions defined in the script
            for key, value in combined_scope.items():
                if callable(value) and not isinstance(value, type):  # If the value is a function
                    try:
                        value.__globals__.update(combined_scope)  # Update its global scope
                    except Exception as e:
                        logging.error("Test:Poster:", repr(e))

            algorithm = combined_scope['select']
            func_to_call = Resource.get_resource_function(
                "resource.poster.sample", ResourceType.DATA
            )
            path_to_images = func_to_call()
            result = algorithm(path_to_images)
            for pic in result:
                src_path = pic['path']
        except Exception as e:
            self._test_result = False
            self._error_msg = self._user_msg = f"Test:Poster: Algorithm could not be checked due to the following error: {repr(e)}"

        return self._test_result


