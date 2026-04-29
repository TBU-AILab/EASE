<<<<<<< HEAD
from fopimt.task_dto import TaskExecutionContext
from fopimt.utils.render_utils import DefaultStatRenderer

=======
from ..loader import Parameter
>>>>>>> origin/main
from ..modul import Modul
from ..modul_dto import StatResult
from ..solutions.solution import Solution


class Stat(Modul):
    """
    Parent statistical module.
    """

    def _init_params(self):
        super()._init_params()

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate_statistic(
        self,
        solutions: list[Solution],
        task_execution_context: TaskExecutionContext,
    ) -> StatResult:
        """
        Statistical evaluation of the list of Solutions.
        :param solutions: List of Solutions which should be evaluated.
        :return: StatResult object containing the statistical results.
        """
        raise NotImplementedError("Function needs to be implemented")

    def export(self, path: str) -> None:
        """
        Export the evaluated statistics. The path must contain the name of the dir for export.
        :param path: Dirname for the export.
        :return: None
        """
        raise NotImplementedError("Function needs to be implemented")

    @staticmethod
    def render_html(
        modul_result: StatResult,
        task_execution_context: TaskExecutionContext,
        output_dir: str,
    ) -> str:
        """
        Returns HTML representation of the evaluation. Used for visualization.
        :return: HTML string
        """
        return DefaultStatRenderer.render_template(
            modul_result,
            output_format="html",
        )

    @staticmethod
    def render_latex(
        modul_result: StatResult,
        task_execution_context: TaskExecutionContext,
        output_dir: str,
    ) -> str:
        """
        Returns LaTeX representation of the evaluation. Used for visualization.
        :return: LaTeX string
        """
        return DefaultStatRenderer.render_template(
            modul_result,
            output_format="latex",
        )

    ####################################################################
    #########  Private functions
    ####################################################################
