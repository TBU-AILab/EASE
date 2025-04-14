import copy
import os.path

from ..loader import Parameter, Loader, PackageType, PrimitiveType
from .analysis import Analysis
from ..solutions.solution import Solution
from ..message import Message


class AnalysisLLM(Analysis):

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        llms = Loader((PackageType.LLMConnector,)).get_package(PackageType.LLMConnector).get_moduls()
        return {
            'llm_prompt': Parameter(short_name="llm_prompt", type=PrimitiveType.markdown,
                                    long_name="LLM prompt",
                                    description="Prompt message for LLM",
                                    default="Analyze the provided data."
                                    ),

            'llm': Parameter(short_name='llm', type=PrimitiveType.enum,
                             long_name='LLM', description='LLM connector',
                             enum_options=llms),
        }

    def _init_params(self):
        super()._init_params()
        self.llm = self.parameters.get('llm', dict())
        self._llmconnector = Loader().get_package(PackageType.LLMConnector).get_modul_imported(self.llm['short_name'])(self.llm['parameters'])
        self._llm_prompt = self.parameters.get("llm_prompt", self.get_parameters().get('llm_prompt').default)
        self._feedback = "There is no feedback."


    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate_analysis(self, solution: Solution) -> None:
        """
        Capture the state of the solution for analysis.
        :param solution: Instance of the Solution.
        :return: None
        """
        # get LLM feedback
        data = solution.get_input()
        if 'image' in solution.get_tags()['output']:
            msg = Message(role=self._llmconnector.get_role_user(), message=self._llm_prompt)
            msg.set_metadata(label='image', data=data)
        else:
            msg = Message(role=self._llmconnector.get_role_user(), message=f"{self._llm_prompt}\n{data}")

        msg_response = self._llmconnector.send([msg])

        # set the feedback from LLM to solution feedback
        feedback = msg_response.get_content()
        self._feedback = feedback


    def export(self, path: str, id: str) -> None:
        """
        Function exports string suitable for console text output.
        :return: String for print() function.
        """

        out = str(self._feedback)
        with open(os.path.join(path, id + "_" + self.get_short_name() + ".txt"), "w", encoding="utf-8") as f:
            f.write(out)

    def get_feedback(self) -> str:
        return str(self._feedback)

    @classmethod
    def get_short_name(cls) -> str:
        return "anal.llm"

    @classmethod
    def get_long_name(cls) -> str:
        return "LLM analysis"

    @classmethod
    def get_description(cls) -> str:
        return "Analysis of the solution via selected LLM."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': set(),
            'output': set()
        }

    ####################################################################
    #########  Private functions
    ####################################################################
