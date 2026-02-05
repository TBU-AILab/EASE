import copy

from .evaluator import Evaluator
from ..solutions.solution import Solution
from ..solutions.solution_image import SolutionImage
from ..llmconnectors.llmconnector import LLMConnector
from ..message import Message
from ..loader import Parameter, PrimitiveType, Loader, PackageType
import re


class EvaluatorMultiLlmFeedback(Evaluator):

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        llms = Loader((PackageType.LLMConnector,)).get_package(PackageType.LLMConnector).get_moduls()

        param_dict = {
            'llms': Parameter(short_name='llms', long_name='LLM connectors', description='A list of LLM connectors '
                                                                                         'with weights.',
                              type=PrimitiveType.list, required=True,
                              default=[
                                  {
                                      'llm': Parameter(short_name='llm', type=PrimitiveType.enum,
                                                       long_name='LLM', description='LLM connector',
                                                       enum_options=llms, required=True),
                                  }
                              ]),

            'feedback_msg_template': Parameter(short_name="feedback_msg_template", type=PrimitiveType.markdown,
                                               long_name="Template for a feedback message",
                                               description="Feedback message for evaluation. Can use {keywords}",
                                               default='{llm_name}:\n{llm_feedback}\n\n"'),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.markdown,
                                           long_name="Template for an initial message",
                                           description="Initial message for evaluation. Specific for each evaluator.",
                                           default="You are a LLM Harry.", readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences", enum_options=['llm_name', 'llm_feedback'], readonly=True)
        }

        return param_dict

    def _init_params(self):
        super()._init_params()
        self._llms = []
        for llm in self.parameters.get('llms', []):
            self._llms.append(
                Loader().get_package(PackageType.LLMConnector).get_modul_imported(llm['llm']['short_name'])(
                    llm['llm']['parameters']))


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
        feedback = ""
        for llm in self._llms:
            if 'image' in solution.get_tags()['output']:
                msg = Message(role=llm.get_role_user(), message="Rate this image on a scale of 0 to 10 (0 the worst, 10 the best).")
                msg.set_metadata(label='image', data=data)
            else:
                msg = Message(role=llm.get_role_user(), message=f"Rate this solution on a scale of 0 to 10 (0 the worst, 10 the best).\n{data}")

            msg_response = llm.send([msg])

            # set the feedback from LLM to solution feedback
            single_feedback = msg_response.get_content()

            keys = {
                'llm_name': llm.get_short_name(),
                'llm_feedback': single_feedback
            }
            feedback += self.get_feedback_msg_template().format(**keys)

        solution.set_feedback(feedback)
        solution.add_metadata(name="feedback", value=feedback)

        # Dummy fitness
        fitness = -1
        solution.set_fitness(fitness)

        self._check_if_best(solution)

        return fitness


    @classmethod
    def get_short_name(cls) -> str:
        return "eval.multillmfeedback"

    @classmethod
    def get_long_name(cls) -> str:
        return "multiLLMfeedback"

    @classmethod
    def get_description(cls) -> str:
        return "LLMs-based evaluator."

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
