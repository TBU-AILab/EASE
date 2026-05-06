import logging
from typing import Optional

import numpy as np
from .evaluator import Evaluator, EvaluatorResult
from ..solutions.solution import Solution
from ..loader import Parameter, PrimitiveType, Loader
import copy
import pandas as pd
from scipy.stats import ranksums
import concurrent.futures
from ..utils.import_utils import dynamic_import
from ..resource.resource import Resource
from ..solutions.solution import Solution
from ..task_dto import TaskExecutionContext, OptimizationGoal
from ..loader_dto import PackageType

from ..resource.metahuristic.metaheuristic_runner import Runner


class EvaluatorPaperContextSummarizer(Evaluator):
    """
    Evaluator for Paper Context Summarize idea. Based on metaheuristic evaluator. The algorithm must be Python code and follow predefined structure
    defined by Runner class.
    :param params: Various parameters for the algorithm.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        llms = (
            Loader((PackageType.LLMConnector,))
            .get_package(PackageType.LLMConnector)
            .get_moduls()
        )
        return {
            "llm": Parameter(
                short_name="llm", type=PrimitiveType.enum, enum_options=llms
            ),
            'feedback_msg_template': Parameter(short_name="feedback_msg_template", type=PrimitiveType.markdown,
                                               long_name="Template for a feedback message",
                                               description="Feedback message for evaluation. Can use {keywords}",
                                               default="{best}"
                                               ),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.markdown,
                                           long_name="Template for an initial message",
                                           description="Initial message for evaluation. Specific for each evaluator.",
                                           default='''Your task is to propose an algorithm to find a set of input parameter values that lead to minimum output value in a limited time. The template for the algorithm is given below. Deliver Python code that is fully operational and self-contained, requiring no external libraries or modifications post-delivery

Glossary:
func - function that returns output value (float) for an array of input parameter values (np.array).
dim - (int) dimension of the input vector.
bounds - (list) specified lower and upper bounds for the input vector values. A pair for each dimension.
max_time - (int) maximum acceptable time in seconds to return a result.

Template:

def run(func, dim, bounds, max_time):

    [Algorithm body]

    # return fitness of the best found solution
    return best

Example implementation of a random search algorithm in the given template:

import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')

    # Algorithm body
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time):
            return best

        params = [np.random.uniform(low, high) for low, high in bounds]
        fitness = func(params)
        if best is None or fitness <= best:
            best = fitness
        ''',
                                           readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences",
                                  enum_options=['last', 'best', 'all',
                                                'last-sum1', 'best-sum1', 'all-sum1',
                                                'last-sum2', 'best-sum2', 'all-sum2'
                                                ],
                                  readonly=True),
            'time': Parameter(short_name="time", type=PrimitiveType.int, long_name='Max time [s]',
                                  description="Max time for evaluation in seconds.", default=5),

        }

    def _init_params(self):
        super()._init_params()

        self.llm = self.parameters.get("llm", dict())
        self._llmconnector = (
            Loader()
            .get_package(PackageType.LLMConnector)
            .get_modul_imported(self.llm["short_name"])(self.llm["parameters"])
        )

        self.function = Resource.get_resource_function('resource.gnbg.f_24', "metabenchmark")()
        self._max_time = self.parameters.get('time', 5)
        self._best = None
        self._solution_history: list = []

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate(self,
                 solution: Solution,
                 opt_goal: OptimizationGoal = OptimizationGoal.MINIMIZATION) -> EvaluatorResult:
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
                        logging.error("Test:Meta:", repr(e))

            algorithm = combined_scope['run']

            df_results = pd.DataFrame(
                columns=['Function', 'Dimension', 'Runs', 'Max_fes', 'Result_fitness', 'Result_params', 'Result_eval',
                         'Min', 'Max', 'Mean', 'Median', 'STD'])
            min_txt = max_txt = mean_txt = median_txt = std_txt = ""
            exceptions = {}

            new_row, exceptions_f = self._process_function(self.function, algorithm)
            fitness = new_row['Min']
            solution.set_fitness(fitness)

            # Add new row to dataframe
            df_results.loc[len(df_results)] = new_row

            # Add exceptions
            exceptions[exceptions_f[0]] = exceptions_f[1]

            solution.add_metadata('results', df_results)
            solution.add_metadata('exceptions', exceptions)

            self._check_if_best(solution)
            self._solution_history.append(solution)

            # Summarization part
            #'last-sum1', 'best-sum1', 'all-sum1',
            #'last-sum2', 'best-sum2', 'all-sum2'
            sum1_llm='''
            
            '''
            sum2_llm='''
            
            '''


            # Update text logs
            min_txt += f"The output value is: {fitness}\n"
            last_txt = f"The output value of the last generated algorithm is: {fitness}\n\n The last generated algorithm code:\n{solution.get_input()}\n"

            index = 1
            self._solution_history.reverse()

            all_txt = "The output values and codes for the last generated algorithms are as follows:\n"
            index = 1
            for sol in self._solution_history:
                all_txt += f"{index}. output value is: {sol.get_fitness()}\n\n {index}. algorithm code is:\n{sol.get_input()}\n\n"
                index += 1
            self._solution_history.reverse()

            _best_history = copy.deepcopy(self._solution_history)
            _best_history.sort(key=lambda x: x.get_fitness())

            best_txt = f"The output value of the best generated algorithm is: {self._best.get_fitness()}\n\n The best generated algorithm code:\n{self._best.get_input()}\n"

            self._keys = {
                'min': min_txt,
                'last': last_txt,
                'best': best_txt,
                'all': all_txt
            }

            feedback = self.get_feedback_msg_template().format(**self._keys)
            solution.set_feedback(feedback)

        except Exception as e:
            # Return error in feedback and set fitness to -1
            fitness = -1
            solution.set_fitness(fitness)
            feedback = f"Error during solution evaluation: {repr(e)}\n. Try to fix it."
            solution.set_feedback(feedback)
            logging.error('Evaluator:Metaheuristic: Error during Task evaluation: ' + repr(e))
            return fitness
            #raise e

        return fitness

    @classmethod
    def get_short_name(cls) -> str:
        return "eval.papercontextsummarizer"

    @classmethod
    def get_long_name(cls) -> str:
        return "Paper context summarizer"

    @classmethod
    def get_description(cls) -> str:
        return "Evaluator for Paper Context Summarizer idead. Based on metaheuristic evaluator. Assuming generated Python code with " \
               "template code of the runner."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'python'},
            'output': {'metaheuristic'}
        }

    ####################################################################
    #########  Private functions
    ####################################################################

    def _process_run(self, algorithm, run, func, dim, max_fes, max_time):

        exception_array = []

        logging.info('Evaluator:Metaheuristic:Run' + str(run) + ':F' + str(func) + ':D' + str(dim))
        a = Runner(copy.deepcopy(algorithm), copy.deepcopy(func), dim, func.get_bounds(), max_fes, max_time)
        result = a.run()

        # Append exceptions
        exception_array.append({str(run) + ':maxtimeexception': result.get('maxtimeexception', False)})
        exception_array.append({str(run) + ':maxevalexception': result.get('maxevalexception', False)})
        exception_array.append({str(run) + ':dimexception': result.get('dimexception', False)})
        exception_array.append({str(run) + ':unexpectedexception': result.get('unexpectedexception', False)})
        exception_array.append({str(run) + ':outofboundsexception': result.get('outofboundsexception', False)})

        return result['best'], exception_array

    def _process_function(self, fdict, algorithm):

        func = fdict['func']
        dim = fdict['dim']
        runs = fdict['runs']
        max_fes = fdict['max_fes']
        max_time = self._max_time

        result_array = []
        exception_array = []

        for run in range(runs):
            result, exceptions_a = self._process_run(algorithm, run, func, dim, max_fes, max_time)
            result_array.append(result)
            for exc in exceptions_a:
                exception_array.append(exc)

        exceptions = [str(func), exception_array]

        fitness = [res['fitness'] for res in result_array]
        params = [res['params'] for res in result_array]
        evals = [res['eval_num'] for res in result_array]

        new_row = {
            'Function': str(func),
            'Dimension': dim,
            'Runs': runs,
            'Max_fes': max_fes,
            'Result_fitness': fitness,
            'Result_params': params,
            'Result_eval': evals,
            'Min': min(fitness),
            'Max': max(fitness),
            'Mean': np.mean(fitness),
            'Median': np.median(fitness),
            'STD': np.std(fitness)
        }

        return new_row, exceptions


    def _check_if_best(self, solution: Solution) -> bool:
        """
        Internal function. Compares fitness of saved solution (_best) against parameter solution.
        Saves the best one to the _best.
        Arguments:
            solution: Solution  -- Solution that will be compared and potentially stored.
        """

        if self._best is None:
            self._best = copy.deepcopy(solution)
            #self._keys['best'] = self._best.get_input()
            return True

        if 'results' not in solution.get_metadata().keys() or 'results' not in self._best.get_metadata().keys():
            logging.warning('Evaluator:Metaheuristic: "Could not compare algorithms due to no existing results."')

        if solution.get_fitness() <= self._best.get_fitness():
            self._best = copy.deepcopy(solution)
            # TODO rework, maybe
            #self._keys['best'] = self._best.get_input()
            return True
        return False