from ..modul import Modul, Parameter
#from ..loader import Modul, Parameter
from ..solutions.solution import Solution


class Analysis(Modul):

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate_analysis(self, solution: Solution) -> None:
        """
        Analyse the Solution.
        :param solution: Instance of the Solution.
        :return: None
        """
        raise NotImplementedError("Function needs to be overridden by child class")

    def export(self, path: str, id: str) -> None:
        """
        Export the analyzed solution. The path must contain the name of the dir.
        :param path: Dirname for the export.
        :param id: Filename for the export.
        :return: None
        """
        raise NotImplementedError("Function needs to be implemented")

    def get_feedback(self) -> str:
        """
        If needed. The child class may override this to get feedback back into loop.
        """
        return ""
    ####################################################################
    #########  Private functions
    ####################################################################
