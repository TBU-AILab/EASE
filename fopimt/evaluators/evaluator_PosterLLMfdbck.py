import copy
import logging
import os

from .evaluator import Evaluator
from ..solutions.solution import Solution
from ..message import Message
from ..loader import Parameter, PrimitiveType, Loader, PackageType
import re
from ..resource.resource import Resource, ResourceType
from ..utils.import_utils import dynamic_import


class EvaluatorLlmFeedback(Evaluator):

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        llms = Loader((PackageType.LLMConnector,)).get_package(PackageType.LLMConnector).get_moduls()
        datasets = Resource.get_resources(ResourceType.DATA)
        return {
            'feedback_msg_template': Parameter(short_name="feedback_msg_template", type=PrimitiveType.str,
                                               long_name="Template for a feedback message",
                                               description="Feedback message for evaluation. Can use {keywords}",
                                               default="You are a master python coder. Review the last proposed "
                                                       "solution and give your feedback. What can be improved, "
                                                       "what can be removed and what can be added. Rate the "
                                                       "output on a scale from 1 to"
                                                       "10. 1 being the worst you've ever seen and 10 "
                                                       "being the best one. Also give an explanation of your "
                                                       "rating and potential advice for improvement. The "
                                                       "template for the rating is the"
                                                       "following:\nRating: {value}\nExplanation: {"
                                                       "explanation}\nAdvice: {advice} Please fill in {value}, {"
                                                       "explanation} and {advice} fields."
                                               ),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.str,
                                           long_name="Template for an initial message",
                                           description="Initial message example.",
                                           default="""Generate a Python algorithm that selects a subset of 5 images 
                                           from a folder of TV show frames to be used as promotional posters. The tv 
                                           show is a soap opera, so the algorithm should focus on the specific 
                                           elements of soap operas.

The output must be a list of dictionaries, where each dictionary contains:

'path': the file path of the image.
'rating': a value between 0 and 1 representing the suitability of the image for a poster, where 1 is the most suitable.

The algorithm should use the following template:

import os
import cv2
import numpy as np
import PIL
import tensorflow as tf
import dlib
import face_recognition
import skimage

# Import other relevant libraries based on the approach.

def select(folder_path):
    # Your algorithm goes here

    return selected_images

Notes: Use a combination of image processing libraries, such as OpenCV, Pillow, scikit-image, dlib, 
or face_recognition depending on the genre. Select only 5 images each time the algorithm runs, and provide a rating 
for each (in the range 0-1). The rating should reflect how well the image meets the criteria for a promotional 
poster, which may differ by genre. Do not include any example usage or direct method calls. Only the implementation 
of the function is required.""",
                                           readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences",
                                  enum_options=[],
                                  readonly=True),
            'dataset': Parameter(short_name='dataset', type=PrimitiveType.enum,
                                 enum_options=datasets['short_names'],
                                 enum_long_names=datasets['long_names'],
                                 enum_descriptions=datasets['descriptions']
                                 ),
            'llm': Parameter(short_name='llm', type=PrimitiveType.enum,
                             enum_options=llms),
        }

    def _init_params(self):
        super()._init_params()
        self.llm = self.parameters.get('llm', dict())
        self._llmconnector = Loader().get_package(PackageType.LLMConnector).get_modul_imported(self.llm['short_name'])(
            self.llm['parameters'])
        self._dataset = self.parameters.get('dataset',
                                            self.get_parameters().get('dataset').default)
        if self.parameters.get('dataset'):
            func_to_call = Resource.get_resource_function(
                self.parameters.get('dataset'), ResourceType.DATA
            )
            self.path = func_to_call()
        self._feedback_prompt = self.parameters.get("feedback_msg_template",
                                                    self.get_parameters().get('feedback_msg_template').default)

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

        local_scope = {}

        # Dynamic import of solution-specific libraries
        exec_globals = {}
        if 'modules' in solution.get_metadata().keys():
            imports = solution.get_metadata()['modules']
            for module_name, specific_part, alias in imports:
                dynamic_import(module_name, specific_part, alias, exec_globals)

        try:
            compile(solution.get_input(), "solution_to_evaluate.py", "exec")
            exec(solution.get_input(), exec_globals, local_scope)

            # Merge exec_globals and exec_locals to ensure functions can access each other
            combined_scope = {**exec_globals, **local_scope}

            # Rebind the global scope for all functions defined in the script
            for key, value in combined_scope.items():
                if callable(value) and not isinstance(value, type):  # If the value is a function
                    try:
                        value.__globals__.update(combined_scope)  # Update its global scope
                    except Exception as e:
                        logging.error("Test:Poster:", repr(e))

            algorithm = combined_scope['select']
            path_to_images = self.path
            result = algorithm(path_to_images)

            solution.add_metadata('data', result)
            base_path = os.path.dirname(solution.get_path())

            for pic in result:
                src_path = pic['path']
                file_name = os.path.basename(src_path)
                dst_path = os.path.join(base_path, file_name)
                self._copy_file(src_path, dst_path)

            #print(result)

        except Exception as e:
            logging.error('Evaluator:Poster: Error during Task evaluation: ' + repr(e))
            raise e

        return fitness

    @classmethod
    def get_short_name(cls) -> str:
        return "eval.posterllmfdbck"

    @classmethod
    def get_long_name(cls) -> str:
        return "Poster plus LLM feedback"

    @classmethod
    def get_description(cls) -> str:
        return "Poster evaluator plus LLM-based evaluator giving text feedback but also a fitness value from the specified range."

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


    def _copy_file(self, source_path, destination_path):
        try:
            if os.path.isfile(source_path):
                with open(source_path, 'rb') as src_file:
                    with open(destination_path, 'wb') as dst_file:
                        # Read and write the file in chunks
                        while chunk := src_file.read(1024):  # Adjust chunk size if needed
                            dst_file.write(chunk)
        except Exception as e:
            logging.error("Evaluator:Poster: Error occurred while copying file:" + repr(e))
