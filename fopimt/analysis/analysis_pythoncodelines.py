import copy
import os.path

from ..loader import Parameter
from .analysis import Analysis
from ..solutions.solution import Solution


class AnalysisPythonCodeLines(Analysis):

    def _init_params(self):
        self._lines: int = 0

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate_analysis(self, solution: Solution) -> None:
        """
        Capture the state of the solution for analysis.
        :param solution: Instance of the Solution.
        :return: None
        """
        # TODO assuming that Solution is of a type python code... This should be checked somehow
        if not isinstance(solution.get_input(), str):
            raise TypeError('AnalysisPythonCodeLines: The solution is not a string')

        count = 0
        for line in solution.get_input().split('\n'):
            linet = line.lstrip()
            if linet == '':
                continue
            if not linet.startswith('#'):
                count += 1
        self._lines = count

    def export(self, path: str, id: str) -> None:
        """
        Function exports string suitable for console text output.
        :return: String for print() function.
        """

        out = "Code analysis: How many lines of code were generated?\n"
        out += str(self._lines)
        with open(os.path.join(path, id+"_"+self.get_short_name()+".txt"), "w", encoding="utf-8") as f:
            f.write(out)


    @classmethod
    def get_short_name(cls) -> str:
        return "anal.pcodelines"

    @classmethod
    def get_long_name(cls) -> str:
        return "Python code lines"

    @classmethod
    def get_description(cls) -> str:
        return "Analysis of the number of code lines generated in Python language."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'python'},
            'output': set()
        }

    ####################################################################
    #########  Private functions
    ####################################################################
