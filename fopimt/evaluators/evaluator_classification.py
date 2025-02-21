import logging

import numpy as np
import pandas as pd
import copy

from ..resource.resource import Resource, ResourceType
from .evaluator import Evaluator
from ..modul import Modul
from ..solutions.solution import Solution
from ..utils.import_utils import dynamic_import
from ..loader import Parameter, PrimitiveType

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


class EvaluatorClassification(Evaluator):
    """
    Evaluator for classification algorithms. The algorithm must be Python code and follow predefined structure
    defined in TODO.
    :param params: Various parameters for the algorithm.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        datasets = Resource.get_resources(ResourceType.CLASSIFICATION)
        return {
            'feedback_msg_template': Parameter(short_name="feedback_msg_template", type=PrimitiveType.str,
                                               long_name="Template for a feedback message",
                                               description="Feedback message for evaluation. Can use {keywords}",
                                               default="The metrics of the proposed model are:\n{accuracy}{recall}{"
                                                       "f1-score}{errors}"
                                               ),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.str,
                                           long_name="Template for an initial message",
                                           description="Initial message for evaluation. Specific for each evaluator.",
                                           default="Your task as an advanced AI is to innovate in the design of a classification algorithms. \n        You are encouraged to be inventive and experiment with various strategies, including adapting existing \n        algorithms or combining them to form new methodologies. Do not include any testing functions or statistical \n        tests, as these are conducted externally. Ensure that you adhere strictly to the provided structure: method \n        name and attributes are given. Focus on developing the [Algorithm body] of the \"predict\" method. Your \n        innovative solution should be fully functional within this framework, without requiring external libraries \n        other than numpy. Here is the template you need to fill followed by an example.\n\nTemplate:\n\nimport numpy as np\ndef predict(X_train, y_train, X_test):\n\n    [Algorithm body]\n\n    return predictions\n\nExample implementation of a kNN algorithm in the given template:\n\nimport numpy as np\ndef predict(X_train, y_train, X_test):\n\n    k = 5\n    predictions = []\n    for x in X_test:\n        distances = np.linalg.norm(X_train - x, axis=1)\n        k_indices = np.argsort(distances)[:k]\n        k_nearest_labels = y_train[k_indices]\n        prediction = np.bincount(k_nearest_labels).argmax()\n        predictions.append(prediction)\n\n    return predictions",
                                           readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences",
                                  enum_options=['accuracy', 'precision', 'recall', 'f1-score', 'errors'], readonly=True),

            'error_limit': Parameter(short_name='error_limit', description="Number of reported wrong classifications",
                                     type=PrimitiveType.int, min_value=-1, default=-1),

            'dataset': Parameter(short_name="dataset", type=PrimitiveType.enum,
                                   enum_options=datasets['short_names'],
                                   enum_long_names=datasets['long_names'],
                                   enum_descriptions=datasets['descriptions']
                                   ),

        }

    def _init_params(self):
        super()._init_params()
        if self.parameters.get('dataset'):
            func_to_call = Resource.get_resource_function(
                self.parameters.get('dataset'), ResourceType.CLASSIFICATION
            )
            self._dataset = func_to_call()

        self._train = self._dataset['train_data']
        self._test = self._dataset['test_data']

        # # Limit for number of reported errors for the feedback
        self._error_limit = self.parameters.get('error_limit', self.get_parameters().get('error_limit').default)
        self._rnd = np.random.default_rng()

        if self._train is None or self._test is None:
            logging.error("Train and test sets have to be specified.")



    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate(self, solution: Solution) -> float:
        """
        Evaluation function. Returns quality of solution as float number.
        Arguments:
            solution: Solution  -- Solution that will be evaluated.
        """

        fitness = 0

        local_scope = {}

        # Dynamic import of solution-specific libraries
        exec_globals = {}
        if 'modules' in solution.get_metadata().keys():
            imports = solution.get_metadata()['modules']
            for module_name, specific_part, alias in imports:
                dynamic_import(module_name, specific_part, alias, exec_globals)

        errors_txt = "Wrong classifications:\n"
        errors = []
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
                        logging.error("Eval:Classification:", repr(e))

            predict = combined_scope['predict']

            y_pred = predict(self._train['X'], self._train['y'], self._test['X'])
            y_true = self._test['y']

            for i in range(len(self._test['X'])):
                if y_true[i] != y_pred[i]:
                    errors.append({
                        'x': (self._test['X'][i]).tolist(),
                        't': y_true[i],
                        'y': y_pred[i]
                    })

            # Limit errors for the feedback msg
            if self._error_limit != -1 and len(errors) > self._error_limit:
                errors_limited = self._rnd.choice(errors, self._error_limit)
            else:
                errors_limited = errors

            for err in errors_limited:
                errors_txt += f"input: {err['x']}, target: {err['t']}, prediction: {err['y']}\n"

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')

            results = {
                'accuracy': accuracy,
                'recall': recall,
                'f1-score': f1,
                'precision': precision,
                'errors': errors
            }

            acc_txt = f"The accuracy score of the model is = {accuracy}\n"
            prec_txt = f"The precision score of the model is = {precision}\n"
            recall_txt = f"The recall score of the model is = {recall}\n"
            f1_txt = f"The f1-score of the model is = {f1}\n"

            solution.add_metadata('results', results)

            self._keys = {
                'accuracy': acc_txt,
                'precision': prec_txt,
                'recall': recall_txt,
                'f1-score': f1_txt,
                'errors': errors_txt
            }

            feedback = self.get_feedback_msg_template().format(**self._keys)
            solution.set_feedback(feedback)

            solution.set_fitness(accuracy)
            fitness = accuracy

            self._check_if_best(solution)

        except Exception as e:
            logging.error('Evaluator:Classification: Error during Task evaluation: ' + repr(e))
            raise e

        return fitness



    @classmethod
    def get_short_name(cls) -> str:
        return "eval.class"

    @classmethod
    def get_long_name(cls) -> str:
        return "Classification"

    @classmethod
    def get_description(cls) -> str:
        return "Evaluator for classification algorithms."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'python'},
            'output': {'classification'}
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
        else:
            return False
