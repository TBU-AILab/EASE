from ..modul import Modul
from ..solutions.solution import Solution
from ..loader import Parameter


class Stat(Modul):
    """
    Parent statistical module.
    """
    def _init_params(self):
        super()._init_params()

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate_statistic(self, solutions: list[Solution]) -> None:
        """
        Statistical evaluation of the list of Solutions.
        :param solutions: List of Solutions which should be evaluated.
        :return: None
        """
        raise NotImplementedError("Function needs to be implemented")

    def export(self, path: str) -> None:
        """
        Export the evaluated statistics. The path must contain the name of the file.
        :param path: Filename for the export.
        :return: None
        """
        raise NotImplementedError("Function needs to be implemented")

    ####################################################################
    #########  Private functions
    ####################################################################
