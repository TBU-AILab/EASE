from .test import Test
from ..solutions.solution import Solution
from ..utils.import_utils import dynamic_install

import ast


class TestImportsPython(Test):
    """
    Test designed for correct Python syntax.
    """
    
    def _init_params(self):
        super()._init_params()
        self._error_msg = "Test:PythonImports: OK"
        self._user_msg = "Test:PythonImports: OK"
        self._error_msg_template = \
            "Unable to import or install specified modules. This is the reason: {0}."
        self._user_msg_template = \
            "Unable to import or install specified modules. This is the reason: {0}."

    ####################################################################
    #########  Public functions
    ####################################################################
    def test(self, solution: Solution) -> bool:
        """
        This function exports all imported modules from the python script in solution
        :param solution: Solution to test
        :return: Returns test result (True = test was OK, False = test was not OK)
        """
        self._result = True
        try:
            libraries = self._extract_imports(solution.get_input())
            solution.add_metadata(name="modules", value=libraries)

            for lib in libraries:
                dynamic_install(lib[0])

        except Exception as e:
            self._result = False
            self._error_msg = self._error_msg_template.format(repr(e))
            self._user_msg = self._user_msg_template.format(repr(e))

        return self._result


    @classmethod
    def get_short_name(cls) -> str:
        return "test.pimports"

    @classmethod
    def get_long_name(cls) -> str:
        return "Python imports test"

    @classmethod
    def get_description(cls) -> str:
        return "Exports imported modules from the python script and saves them into solution metadata."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'python'},
            'output': set()
        }

    ####################################################################
    #########  Private functions
    ####################################################################
    def _extract_imports(self, script_content):
        tree = ast.parse(script_content)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Append the module name and alias (if any)
                    imports.append((alias.name, None, alias.asname))  # Whole module import
            elif isinstance(node, ast.ImportFrom):
                # 'module' is the library, 'names' is the specific part(s) being imported
                for alias in node.names:
                    imports.append((node.module, alias.name, alias.asname))

        return imports