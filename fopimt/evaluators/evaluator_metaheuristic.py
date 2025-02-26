import logging

import numpy as np
from .evaluator import Evaluator
from ..solutions.solution import Solution
from ..loader import Parameter, PrimitiveType
import copy
import pandas as pd
from scipy.stats import ranksums
import concurrent.futures
from ..utils.import_utils import dynamic_import
from ..resource.resource import Resource, ResourceType

from ..resource.metahuristic.metaheuristic_runner import Runner


class EvaluatorMetaheuristic(Evaluator):
    """
    Evaluator for metaheuristic algorithms. The algorithm must be Python code and follow predefined structure
    defined by Runner class.
    :param params: Various parameters for the algorithm.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        benchmarks = Resource.get_resources(ResourceType.METABENCHMARK)
        return {
            'feedback_msg_template': Parameter(short_name="feedback_msg_template", type=PrimitiveType.markdown,
                                               long_name="Template for a feedback message",
                                               description="Feedback message for evaluation. Can use {keywords}",
                                               default="The mean results of the tested functions are:\n{mean}\nand the "
                                                       "statistic result is:\n{stats}"
                                               ),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.markdown,
                                           long_name="Template for an initial message",
                                           description="Initial message for evaluation. Specific for each evaluator.",
                                           default='''Your task as an advanced AI is to innovate in the design of a single-objective 
        metaheuristic algorithm aimed at minimizing the objective function. You are encouraged to be inventive and 
        experiment with various strategies, including adapting existing algorithms or combining them to form new 
        methodologies. Do not include any testing functions or statistical tests, as these are conducted externally. 
        Ensure that you adhere strictly to the provided structure: method name and attributes are given. Focus on 
        developing the [Algorithm body] of the "run" method. Expand the search method with your novel approach, 
        ensuring that every aspect of the algorithm's execution is geared towards effectively minimizing the 
        objective function. Your innovative solution should be fully functional within this framework, 
        without requiring external libraries. Here is the template you need to fill followed by an example.

Template:

def run(func, dim, bounds, max_evals):

    [Algorithm body]

    # return fitness of the best found solution
    return best

Example implementation of a random search algorithm in the given template:

import numpy as np

def run(func, dim, bounds, max_evals):
    best = float('inf')

    # Algorithm body
    for eval in range(max_evals):
        params = np.array([np.random.uniform(low, high) for low, high in bounds])
        fitness = func(params)
        if best is None or fitness <= best:
            best = fitness

    return best
        ''',
                                           readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences",
                                  enum_options=['min', 'max', 'mean', 'median', 'std', 'metadata'],
                                  readonly=True),

            'benchmark': Parameter(short_name="benchmark", type=PrimitiveType.enum,
                                   enum_options=benchmarks['short_names'],
                                   enum_long_names=benchmarks['long_names'],
                                   enum_descriptions=benchmarks['descriptions']
                                   ),

            'stats_txt_base': Parameter(short_name="stats_txt_base",
                                        type=PrimitiveType.markdown,
                                        long_name="Text for base statistics",
                                        description="This text specifies how the statistics will be introduced in the "
                                                    "feedback.",
                                        default="The result of the Wilcoxon rank sum test (alpha=0.05) between "
                                                "current and best-so-far solution is the following:\n"
                                        ),
            'stats_txt_better_solution': Parameter(short_name="stats_txt_better_solution",
                                                   type=PrimitiveType.markdown,
                                                   long_name="Text for better solution",
                                                   description="Text representation when a significantly better solution has been found.",
                                                   default="The current solution is significantly better on function {f} dim {"
                                                           "dim} with p-value={p_value}\n"
                                                   ),
            'stats_txt_worse_solution': Parameter(short_name="stats_txt_worse_solution",
                                                  type=PrimitiveType.markdown,
                                                  long_name="Text for worse solution",
                                                  description="Text representation when a significantly worse "
                                                              "solution has been found.",
                                                  default="The best-so-far solution is significantly better on "
                                                          "function {f} dim {dim} with p-value={p_value}\n"
                                                  ),
            'stats_txt_equal_solution': Parameter(short_name="stats_txt_equal_solution",
                                                  type=PrimitiveType.markdown,
                                                  long_name="Text for equal solution",
                                                  description="Text representation when an equally good solution has "
                                                              "been found.",
                                                  default="There is no significant difference between current and "
                                                          "best-so-far solutions on function {f} dim {dim} with "
                                                          "p-value={p_value}\n"
                                                  )
        }

    def _init_params(self):
        super()._init_params()
        if self.parameters.get('benchmark'):
            func_to_call = Resource.get_resource_function(
                self.parameters.get('benchmark'), ResourceType.METABENCHMARK
            )
            self.functions = func_to_call()

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate(self, solution: Solution) -> float:
        """
        Evaluation function. Returns quality of solution as float number.
        Arguments:
            solution: Solution  -- Solution that will be evaluated.
        """
        if self._best is not None:
            fitness = self._best.get_fitness() + 1
        else:
            fitness = 0
        solution.set_fitness(fitness)

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

            for fdict in self.functions:
                new_row, exceptions_f = self._process_function(fdict, algorithm)

                # Add new row to dataframe
                df_results.loc[len(df_results)] = new_row

                # Add exceptions
                exceptions[exceptions_f[0]] = exceptions_f[1]

                # Update text logs
                min_txt += f"best result of function {new_row['Function']} dim {new_row['Dimension']} = {new_row['Min']}\n"
                max_txt += f"worst result of function {new_row['Function']} dim {new_row['Dimension']} = {new_row['Max']}\n"
                mean_txt += f"mean result of function {new_row['Function']} dim {new_row['Dimension']} = {new_row['Mean']}\n"
                median_txt += f"median result of function {new_row['Function']} dim {new_row['Dimension']} = {new_row['Median']}\n"
                std_txt += f"standard deviation of results of function {new_row['Function']} dim {new_row['Dimension']} = {new_row['STD']}\n"

            solution.add_metadata('results', df_results)
            solution.add_metadata('exceptions', exceptions)
            self._keys = {
                'min': min_txt,
                'max': max_txt,
                'mean': mean_txt,
                'median': median_txt,
                'std': std_txt,
                'metadata': df_results,
                'stats': 'not enough data'
            }

            self._check_if_best(solution)

            feedback = self.get_feedback_msg_template().format(**self._keys)
            solution.set_feedback(feedback)

        except Exception as e:
            logging.error('Evaluator:Metaheuristic: Error during Task evaluation: ' + repr(e))
            raise e

        return fitness

    @classmethod
    def get_short_name(cls) -> str:
        return "eval.meta"

    @classmethod
    def get_long_name(cls) -> str:
        return "Metaheuristic"

    @classmethod
    def get_description(cls) -> str:
        return "Evaluator for metaheuristic algorithms. Assuming generated Python code with " \
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

    def _process_run(self, algorithm, run, func, dim, max_fes):

        exception_array = []

        logging.info('Evaluator:Metaheuristic:Run' + str(run) + ':F' + str(func) + ':D' + str(dim))
        a = Runner(copy.deepcopy(algorithm), copy.deepcopy(func), dim, func.get_bounds(), max_fes)
        result = a.run()

        # Append exceptions
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

        result_array = []
        exception_array = []

        for run in range(runs):
            result, exceptions_a = self._process_run(algorithm, run, func, dim, max_fes)
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

    def _process_function_parallel(self, fdict, algorithm):
        # TODO - check if it actually does it in parallel

        func = fdict['func']
        dim = fdict['dim']
        runs = fdict['runs']
        max_fes = fdict['max_fes']

        result_array = []
        exception_array = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_run, algorithm, run, func, dim, max_fes) for run in range(runs)]

            for future in concurrent.futures.as_completed(futures):
                result, exceptions_a = future.result()

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
            return True

        if 'results' not in solution.get_metadata().keys() or 'results' not in self._best.get_metadata().keys():
            logging.warning('Evaluator:Metaheuristic: "Could not compare algorithms due to no existing results."')

        score = 0

        df_current = solution.get_metadata()['results']
        df_best = self._best.get_metadata()['results']

        # Group by the matching columns
        grouped1 = df_current.groupby(['Function', 'Dimension', 'Runs', 'Max_fes'])
        grouped2 = df_best.groupby(['Function', 'Dimension', 'Runs', 'Max_fes'])

        stats_txt = self.parameters.get('stats_txt_base', self.get_parameters().get('stats_txt_base').default)
        # Iterate through the groups
        for key, group1 in grouped1:
            if key in grouped2.groups:
                group2 = grouped2.get_group(key)

                # Extract the arrays from Result_fitness
                fitness1 = group1['Result_fitness'].values[0]
                fitness2 = group2['Result_fitness'].values[0]

                # Perform Mann-Whitney U test
                stat, p_value = ranksums(fitness1, fitness2)

                # Determine the better algorithm
                if p_value < 0.05:
                    if stat < 0:
                        score += 1

                        stats_txt += self.parameters.get('stats_txt_better_solution', self.get_parameters().get('stats_txt_better_solution').default).format(
                            f=key[0], dim=key[1], p_value=p_value)
                    else:
                        score -= 1
                        stats_txt += self.parameters.get('stats_txt_worse_solution', self.get_parameters().get('stats_txt_worse_solution').default).format(
                            f=key[0], dim=key[1], p_value=p_value)
                else:
                    stats_txt += self.parameters.get('stats_txt_equal_solution', self.get_parameters().get('stats_txt_equal_solution').default).format(
                        f=key[0], dim=key[1], p_value=p_value)

        self._keys['stats'] = stats_txt
        # non-negative score means better solution
        solution.set_fitness(score)
        if score >= 0:
            self._best = copy.deepcopy(solution)
            return True
        return False
