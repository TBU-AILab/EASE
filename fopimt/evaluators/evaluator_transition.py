import logging

import numpy as np
import copy
import pickle

from ..resource.resource import Resource, ResourceType
from .evaluator import Evaluator
from ..solutions.solution import Solution
from ..utils.import_utils import dynamic_import
from ..loader import Parameter, PrimitiveType

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


class EvaluatorTransition(Evaluator):
    """
    Evaluator for transition prediction algorithms. The algorithm must be Python code and follow predefined structure
    defined in TODO.
    :param params: Various parameters for the algorithm.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        datasets = Resource.get_resources(ResourceType.CLASSIFICATION)
        return {
            'feedback_msg_template': Parameter(short_name="feedback_msg_template", type=PrimitiveType.markdown,
                                               long_name="Template for a feedback message",
                                               description="Feedback message for evaluation. Can use {keywords}",
                                               default="The metrics of the proposed model are:\n{accuracy}{precision}{recall}{"
                                                       "f1-score}{errors}"
                                               ),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.markdown,
                                           long_name="Template for an initial message",
                                           description="Initial message for evaluation. Specific for each evaluator.",
                                           default="Your task as an advanced AI is to innovate in the design of a classification algorithms.",
                                           readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences",
                                  enum_options=['accuracy', 'precision', 'recall', 'f1-score', 'errors'], readonly=True),

            'error_limit': Parameter(short_name='error_limit', description="Number of reported wrong classifications (-1 for all).",
                                     type=PrimitiveType.int, min_value=-1, default=0),

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
        self._val = self._dataset.get('val_data', None)

        # Limit for number of reported errors for the feedback
        self._error_limit = self.parameters.get('error_limit', self.get_parameters().get('error_limit').default)
        self._rnd = np.random.default_rng(42)

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
                        logging.error("Eval:Transition:", repr(e))

            extract_features = combined_scope['extract_features']
            train = combined_scope['train']
            predict = combined_scope['predict']

            trained_model = train(self._train['X'], self._train['y'], extract_features)
            EvaluatorTransition._pickle_model(trained_model, solution.get_path() + "_model")

            test_results = EvaluatorTransition._test_on_set(trained_model, predict, extract_features, self._test)

            # Limit errors for the feedback msg
            if self._error_limit != -1 and len(errors) > self._error_limit:
                errors_limited = self._rnd.choice(errors, self._error_limit)
            else:
                errors_limited = errors

            for err in errors_limited:
                errors_txt += f"input: {err['x']}, target: {err['t']}, prediction: {err['y']}\n"

            acc_txt = f"The accuracy score of the model is = {test_results['accuracy']} [test set]\n"
            prec_txt = f"The precision score of the model is = {test_results['precision']} [test set]\n"
            recall_txt = f"The recall score of the model is = {test_results['recall']} [test set]\n"
            f1_txt = f"The f1-score of the model is = {test_results['f1-score']} [test set]\n"

            solution.add_metadata('test_results', test_results)

            self._keys = {
                'accuracy': acc_txt,
                'precision': prec_txt,
                'recall': recall_txt,
                'f1-score': f1_txt,
                'errors': errors_txt
            }

            if self._val:
                val_results = EvaluatorTransition._test_on_set(trained_model, predict, extract_features, self._val)
                acc_txt = f"The accuracy score of the model is = {val_results['accuracy']} [val set]\n"
                self._keys['val_accuracy'] = acc_txt
                prec_txt = f"The precision score of the model is = {val_results['precision']} [val set]\n"
                self._keys['val_precision'] = prec_txt
                recall_txt = f"The recall score of the model is = {val_results['recall']} [val set]\n"
                self._keys['val_recall'] = recall_txt
                f1_txt = f"The f1-score of the model is = {val_results['f1-score']} [val set]\n"
                self._keys['val_f1-score'] = f1_txt

                solution.add_metadata('val_results', val_results)

            feedback = self.get_feedback_msg_template().format(**self._keys)
            solution.set_feedback(feedback)

            f1 = test_results['f1-score']
            solution.set_fitness(f1)
            fitness = f1

            self._check_if_best(solution)

        except Exception as e:
            logging.error('Evaluator:Transition: Error during Task evaluation: ' + repr(e))
            raise e

        return fitness


    @classmethod
    def get_short_name(cls) -> str:
        return "eval.transition"

    @classmethod
    def get_long_name(cls) -> str:
        return "Transition"

    @classmethod
    def get_description(cls) -> str:
        return "Evaluator for transition prediction algorithms (ModernTV)."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'python'},
            'output': {'classification'}
        }

    ####################################################################
    #########  Private functions
    ####################################################################
    @staticmethod
    def _pickle_model(model, name):
        file_name = name + ".pkl"

        # save
        pickle.dump(model, open(file_name, "wb"))


    @staticmethod
    def _test_on_set(model, predict, extract_features, the_set) -> dict:

        errors = []
        y_pred = []
        y_true = the_set['y']

        for i in range(len(the_set['X'])):

            pred = predict(model, extract_features, the_set['X'][i])
            y_pred_one = pred['prediction']
            y_pred.append(y_pred_one)

            if y_true[i] != y_pred_one:
                errors.append({
                    'x': the_set['X'][i],
                    't': y_true[i],
                    'y': y_pred_one
                })

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

        return results


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
