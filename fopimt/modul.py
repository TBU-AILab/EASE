from typing import TYPE_CHECKING, Any

from .loader_dto import Parameter

if TYPE_CHECKING:
    from fopimt.task_dto import TaskExecutionContext


class Modul:
    def __init__(self, parameters: dict[str, Any]):
        """
        Basic Modul. All classes in defined packages must inherit it.
        Defines basic description operations for class.
        """
        self.parameters = parameters
        self._init_params()

    ####################################################################
    #########  Public functions
    ####################################################################
    @classmethod
    def get_short_name(cls) -> str:
        raise NotImplementedError("Return short name of the Modul")

    @classmethod
    def get_long_name(cls) -> str:
        raise NotImplementedError("Return long name of the Modul")

    @classmethod
    def get_description(cls) -> str:
        raise NotImplementedError("Return description of the Modul")

    @classmethod
    def get_tags(cls) -> dict:
        raise NotImplementedError("return tags of the Modul")

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        # Maybe overriden by subsequent class
        return dict()

    @classmethod
    def get_order(cls) -> int:
        """Return order of execution in pipeline. Lower number means earlier execution."""
        return 0

    @staticmethod
    def render_html(
        modul_result,
        task_execution_context: "TaskExecutionContext",
        output_dir: str,
    ) -> str:
        """
        Returns HTML representation of the evaluation. Used for visualization.
        :return: HTML string
        """
        raise NotImplementedError("HTML rendering not implemented for this modul")

    @staticmethod
    def render_latex(
        modul_result,
        task_execution_context: "TaskExecutionContext",
        output_dir: str,
    ) -> str:
        """
        Returns LaTeX representation of the evaluation. Used for visualization.
        :return: LaTeX string
        """
        raise NotImplementedError("LaTeX rendering not implemented for this modul")

    ####################################################################
    #########  Private functions
    ####################################################################
    def _init_params(self):
        pass
