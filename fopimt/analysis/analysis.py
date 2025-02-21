from ..modul import Modul, Parameter
#from ..loader import Modul, Parameter
from ..solutions.solution import Solution


class Analysis(Modul):

    ####################################################################
    #########  Public functions
    ####################################################################
    def capture(self, solution: Solution) -> None:
        """
        Capture the state of the solution for analysis.
        :param solution: Instance of the Solution.
        :return: None
        """
        raise NotImplementedError("Function needs to be overridden by child class")

    def pretty(self) -> str:
        """
        Function returns string suitable for console text output.
        :return: String for print() function.
        """
        raise NotImplementedError("Function needs to be overridden by child class")

    ####################################################################
    #########  Private functions
    ####################################################################
