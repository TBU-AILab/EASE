from fopimt.task_dto import TaskExecutionContext
from fopimt.utils.render_utils import DefaultLLMConnectorRenderer

from ..message import Message
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
    def _set_usage(
        self,
        msg: Message,
        input_tokens: int | None = 0,
        output_tokens: int | None = 0,
        cached_input_tokens: int | None = 0,
        cache_write_input_tokens: int | None = 0,
        reasoning_tokens: int | None = 0,
        duration_s: float | None = None,
        calls: int = 1,
        estimated: bool = False,
    ) -> None:
        """
        Store normalized token usage of one send() into the response Message metadata under 'usage'.
        :param input_tokens: All input (prompt) tokens, including cached ones.
        :param output_tokens: All billed output tokens, including reasoning/thinking tokens.
        :param cached_input_tokens: Input tokens served from the provider cache (subset of input_tokens).
        :param cache_write_input_tokens: Input tokens written to the provider cache (subset of input_tokens).
        :param reasoning_tokens: Reasoning/thinking tokens (subset of output_tokens).
        :param duration_s: Wall-clock duration of the whole send() in seconds.
        :param calls: Number of API calls made within send() (e.g. continuations).
        :param estimated: True if counts are estimated rather than reported by the provider.
        """
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        msg.set_metadata(
            "usage",
            {
                "provider": self._type,
                "model": self.get_model(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": int(cached_input_tokens or 0),
                "cache_write_input_tokens": int(cache_write_input_tokens or 0),
                "reasoning_tokens": int(reasoning_tokens or 0),
                "total_tokens": input_tokens + output_tokens,
                "duration_s": duration_s,
                "calls": calls,
                "estimated": estimated,
            },
        )
