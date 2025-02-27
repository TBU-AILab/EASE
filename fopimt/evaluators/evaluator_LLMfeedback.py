import copy

from .evaluator import Evaluator
from ..solutions.solution import Solution
from ..solutions.solution_image import SolutionImage
from ..llmconnectors.llmconnector import LLMConnector
from ..message import Message
from ..loader import Parameter, PrimitiveType, Loader, PackageType
import re


class EvaluatorLlmFeedback(Evaluator):

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        llms = Loader((PackageType.LLMConnector,)).get_package(PackageType.LLMConnector).get_moduls()
        return {
            'feedback_msg_template': Parameter(short_name="feedback_msg_template", type=PrimitiveType.markdown,
                                               long_name="Template for a feedback message",
                                               description="Feedback message for evaluation. Can use {keywords}",
                                               default="Please rate the following output on a scale from 1 to "
                                                              "10. 1 being the worst you've ever seen and 10 "
                                                              "being the best one. Also give an explanation of your "
                                                              "rating and potential advice for improvement. The "
                                                              "template for the rating is the "
                                                              "following:\nRating: {value}\nExplanation: {"
                                                              "explanation}\nAdvice: {advice} Please fill in {value}, {"
                                                              "explanation} and {advice} fields."
                                               ),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.markdown,
                                           long_name="Template for an initial message",
                                           description="Initial message example.",
                                           default="Your task is to draw Mona Lisa-like picture.",
                                           readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences",
                                  enum_options=[],
                                  readonly=True),

            'llm': Parameter(short_name='llm', type=PrimitiveType.enum,
                             enum_options=llms),
        }

    def _init_params(self):
        super()._init_params()
        self.llm = self.parameters.get('llm', dict())
        self._llmconnector = Loader().get_package(PackageType.LLMConnector).get_modul_imported(self.llm['short_name'])(self.llm['parameters'])
        self._feedback_prompt = self.parameters.get("feedback_msg_template", self.get_parameters().get('feedback_msg_template').default)

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate(self, solution: Solution) -> float:
        """
        Evaluation function. Returns quality of solution as float number.
        Arguments:
            solution: Solution  -- Solution that will be evaluated.
        """
        # get LLM feedback
        data = solution.get_input()
        if 'image' in solution.get_tags()['output']:
            msg = Message(role=self._llmconnector.get_role_user(), message=self._feedback_prompt)
            msg.set_metadata(label='image', data=data)
        else:
            msg = Message(role=self._llmconnector.get_role_user(), message=f"{self._feedback_prompt}\n{data}")

        msg_response = self._llmconnector.send([msg])

        # set the feedback from LLM to solution feedback
        feedback = msg_response.get_content()
        solution.set_feedback(feedback)
        solution.add_metadata(name="feedback", value=feedback)

        # Evaluation of the fitness and fitness setting
        if self._best is not None:
            fitness = self._extract_rating(feedback)
        else:
            fitness = 0
        solution.set_fitness(fitness)

        self._check_if_best(solution)

        return fitness


    @classmethod
    def get_short_name(cls) -> str:
        return "eval.llmfeedback"

    @classmethod
    def get_long_name(cls) -> str:
        return "LLMfeedback"

    @classmethod
    def get_description(cls) -> str:
        return "LLM-based evaluator giving text feedback but also a fitness value from the specified range."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': set(),
            'output': set()
        }

    ####################################################################
    #########  Private functions
    ####################################################################
    def _check_if_best(self, solution: Solution) -> bool:
        """
        Internal function. Compares fitness of saved solution (_best) against parameter solution.
        Saves the best one to the _best.
        Arguments:
            solution: Solution  -- Solution that will be compared and potentially stored.
        """
        if self._best is None or solution.get_fitness() >= self._best.get_fitness():
            self._best = copy.deepcopy(solution)
            return True
        return False

    def _extract_rating(self, text: str) -> float:
        # Define a regex pattern to find the rating value
        pattern = r'^Rating:\s*(-?\d+(\.\d+)?)$'

        # Split the text into lines
        lines = text.split('\n')

        # Iterate over each line to find the line with the rating
        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                # Return the rating value as a float
                return float(match.group(1))

        # Return None if the rating is not found or not valid
        return -1
