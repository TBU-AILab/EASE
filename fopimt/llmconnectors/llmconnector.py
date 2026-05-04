from fopimt.task_dto import TaskExecutionContext
from fopimt.utils.render_utils import DefaultLLMConnectorRenderer

from ..modul import Modul
from ..modul_dto import LLMConnectorResult


class LLMConnector(Modul):
    """
    General LLM Connector class.
    Sets low level definitions and provide layer between app and LLM.
    :param short_name: Name of the LLMConnector type. Short version.
    :param long_name: Name of the LLMConnector type. Long version.
    :param description: Description of the LLMConnector type.
    :param tags: Tags associated with the LLMConnector. USed for compatibility checks.
    :param token: Token|ID used for the LLM connector API.
    :param model: Specified model of the LLM.
    """

    def _init_params(self):
        super()._init_params()
        self._type: str | None = None  # Type of LLM (OpenAI, Meta, Google, ...)

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_role_user(self) -> str:
        """
        Get role specification string for USER.
        Returns string.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_role_system(self) -> str:
        """
        Get role specification string for SYSTEM.
        Returns string.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_role_assistant(self) -> str:
        """
        Get role specification string for ASSISTANT.
        Returns string.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def send(self, context) -> LLMConnectorResult:
        """
        Send context to LLM.
        Returns LLMConnectorResult.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_model(self) -> str:
        """
        Returns current LLM model as string.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @staticmethod
    def render_html(
        modul_result: LLMConnectorResult,
        task_execution_context: TaskExecutionContext,
        output_dir: str,
    ) -> str:
        """
        Returns HTML representation of the evaluation. Used for visualization.
        :return: HTML string
        """
        return DefaultLLMConnectorRenderer.render_template(
            modul_result,
            output_format="html",
        )

    @staticmethod
    def render_latex(
        modul_result: LLMConnectorResult,
        task_execution_context: TaskExecutionContext,
        output_dir: str,
    ) -> str:
        """
        Returns LaTeX representation of the evaluation. Used for visualization.
        :return: LaTeX string
        """
        return DefaultLLMConnectorRenderer.render_template(
            modul_result,
            output_format="latex",
        )

    ####################################################################
    #########  Private functions
    ####################################################################
