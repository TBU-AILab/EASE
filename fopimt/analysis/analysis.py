from fopimt.utils.render_utils import DefaultAnalysisRenderer

from ..modul import Modul
from ..modul_dto import AnalysisResult
from ..solutions.solution import Solution
from ..task_dto import TaskExecutionContext


class Analysis(Modul):
    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate_analysis(
        self,
        solution: Solution,
        task_execution_context: TaskExecutionContext,
    ) -> AnalysisResult:
        """
        Analyse the Solution.
        :param solution: Instance of the Solution.
        :param task_execution_context: Instance of the TaskExecutionContext.
        :return: AnalysisResult object containing the analysis results in metadata.
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

    @staticmethod
    def render_html(
        modul_result: AnalysisResult,
        task_execution_context: TaskExecutionContext,
        output_dir: str,
    ) -> str:
        """
        Returns HTML representation of the evaluation. Used for visualization.
        :return: HTML string
        """
        return DefaultAnalysisRenderer.render_template(
            modul_result,
            output_format="html",
        )

    @staticmethod
    def render_latex(
        modul_result: AnalysisResult,
        task_execution_context: TaskExecutionContext,
        output_dir: str,
    ) -> str:
        """
        Returns LaTeX representation of the evaluation. Used for visualization.
        :return: LaTeX string
        """
        return DefaultAnalysisRenderer.render_template(
            modul_result,
            output_format="latex",
        )

    ####################################################################
    #########  Private functions
    ####################################################################
