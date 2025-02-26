from enum import Enum
from typing import Optional
from ..modul import Modul
from ..solutions.solution import Solution
from ..loader import Parameter, PrimitiveType


class Evaluator(Modul):
    """
    General Evaluator class.
    :param params: Dictionary of optional parameters for evaluator.
    :param feedback_msg_template: Template of the message feedback. May contain {KEYWORD}.
    :param init_msg_template: Template of the message init.
    :param short_name: Name of the Evaluator type. Short version.
    :param long_name: Name of the Evaluator type. Long version.
    :param description: Description of the Evaluator type.
    :param tags: Tags associated with the Evaluator. USed for compatibility checks.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            'feedback_msg_template': Parameter(short_name="feedback_msg_template", type=PrimitiveType.markdown,
                                               long_name="Template for a feedback message",
                                               description="Feedback message for evaluation. Can use {keywords}",
                                               default=''),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.markdown,
                                           long_name="Template for an initial message",
                                           description="Initial message for evaluation. Specific for each evaluator.",
                                           default="You are a LLM Harry.", readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences", enum_options=[], readonly=True)
        }

    def _init_params(self):
        super()._init_params()
        self._best: Solution | None = None  # Best Solution found so far
        self._feedback_msg_template = self.parameters.get('feedback_msg_template',
                                                          self.get_parameters().get('feedback_msg_template').default)
        self._feedback_keywords = self.parameters.get('keywords', self.get_parameters().get('keywords').default)
        self._keys: dict = {}
        self._init_msg_template = self.parameters.get('init_msg_template',
                                                      self.get_parameters().get('init_msg_template').default)

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate(self, solution: Solution) -> float:
        """
        Evaluation function. Returns quality of solution as float number.
        Arguments:
            solution: Solution  -- Solution that will be evaluated.
        """
        raise NotImplementedError("Objective function not implemented")

    def get_best(self) -> Solution:
        """
        Returns bets found solution.
        """
        return self._best

    def get_feedback_keywords(self) -> list[str]:
        """
        Returns feedback keywords which may be used in feedback message template.
        :return:
        """
        return self._feedback_keywords

    def get_init_msg_template(self) -> str:
        """
        Returns Template init message.
        :return:
        """
        return self._init_msg_template

    def get_feedback_msg_template(self) -> str:
        """
        Returns Template feedback message.
        :return:
        """
        return self._feedback_msg_template

    ####################################################################
    #########  Private functions
    ####################################################################
