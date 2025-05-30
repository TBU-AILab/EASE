import logging

import numpy as np
import itertools
from .evaluator import Evaluator
from ..solutions.solution import Solution
from ..loader import Parameter, PrimitiveType
import copy
from ..utils.import_utils import dynamic_import
from ..resource.resource import Resource, ResourceType

def _check_inside_triangle(points: np.ndarray) -> bool:
    """
    Returns True if all points are inside or on the boundary of the equilateral triangle
    with vertices (0,0), (1,0), and (0.5, sqrt(3)/2). Returns False otherwise.
    """
    for (x, y) in points:
        if not (
                (y >= 0) and
                (y <= np.sqrt(3) * x) and
                (y <= -np.sqrt(3) * x + np.sqrt(3))
        ):
            return False
    return True


def _triangle_area(a: np.array, b: np.array, c: np.array) -> float:
    return np.abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2


def _evaluate_found_points(points: np.ndarray):

    a = np.array([0, 0])
    b = np.array([1, 0])
    c = np.array([0.5, np.sqrt(3) / 2])

    if _check_inside_triangle(points):
        min_triangle_area = min(
            [_triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)])
        # Normalize the minimum triangle area (since the equilateral triangle is not unit).
        min_area_normalized = min_triangle_area / _triangle_area(a, b, c)
        return min_area_normalized
    else:
        return -1


class EvaluatorHeilbronn11(Evaluator):
    '''
    Evaluator for heilbronn 11 algorithms. The algorithm must be Python code and follow predefined structure.

    def run(fitness_func, check_inside_triangle_func, max_time):
    """
    Parameters:
        - fitness_func: function that accepts an np.ndarray of shape (11, 2) and returns a float
        - check_inside_triangle_func: function that accepts an np.ndarray of shape (11, 2) and returns True/False
        - max_time: time limit for the search in seconds

    Goal:
        - maximization of the fitness
    """

    [Algorithm body]

    # return coordinates and fitness of the best found solution
    return coordinates, fitness

    '''

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        benchmarks = Resource.get_resources(ResourceType.METABENCHMARK)
        return {
            'feedback_msg_template': Parameter(short_name="feedback_msg_template", type=PrimitiveType.markdown,
                                               long_name="Template for a feedback message",
                                               description="Feedback message for evaluation. Can use {keywords}",
                                               default="The best result found by the current solution is:\n{max}\n\nBest-so-far solution:\n{best_solution}\n\nand the "
                                                       "best result is:\n{best_max}"
                                               ),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.markdown,
                                           long_name="Template for an initial message",
                                           description="Initial message for evaluation. Specific for each evaluator.",
                                           default='''Your task as an advanced AI is to design a novel optimization algorithm for the **Heilbronn triangle problem** with $n = 11$ points. The objective is to **maximize a given fitness function**, typically representing the **minimum distance between any pair of 11 points** placed **inside or on the boundary of a triangle** defined by vertices $(0, 0), (1, 0), (0.5, \sqrt{3}/2)$.

You are encouraged to innovate: create a new metaheuristic or adapt and hybridize existing approaches. Do **not** include testing functions, visualization, or statistical analysis — these are handled externally. The optimization process must work by repeatedly generating and improving configurations of 11 two-dimensional points under geometric constraints.

Focus exclusively on filling the `[Algorithm body]` section of the following `run` method template. You must return the best found configuration (`coordinates`) and its corresponding fitness value. The fitness is externally defined and passed into the function. The geometric constraint is checked via `check_inside_triangle_func`, which returns `True` only if all 11 points are valid.

Your algorithm must:

* Operate within a strict time limit (`max_time`, in seconds).
* Use no external libraries (e.g., scikit-learn, DEAP).
* Be fully self-contained and compatible with the template interface.
* Generate valid solutions inside the triangle (use the constraint check function).
* Efficiently search the space to **maximize the fitness function**.
* Only return results when the constraint is satisfied.

Here is the method header and template you need to complete:

```
def run(fitness_func, check_inside_triangle_func, max_time):
    """
    Parameters:
        - fitness_func: function that accepts an np.ndarray of shape (11, 2) and returns a float
        - check_inside_triangle_func: function that accepts an np.ndarray of shape (11, 2) and returns True/False
        - max_time: time limit for the search in seconds
    
    Goal:
        - maximization of the fitness
    """

    [Algorithm body]

    # return coordinates and fitness of the best found solution
    return coordinates, fitness
```

Example implementation of a random search algorithm in the given template:

```
import numpy as np
import time

def run(fitness_func, check_inside_triangle_func, max_time):
    start_time = time.time()
    best_coordinates = None
    best_fitness = -np.inf

    while time.time() - start_time < max_time:
        # Generate 11 random 2D points in bounding box of triangle: x ∈ [0,1], y ∈ [0, sqrt(3)/2]
        candidates = np.random.rand(11, 2)
        candidates[:, 1] *= np.sqrt(3) / 2  # scale y to triangle height

        if check_inside_triangle_func(candidates):
            fitness = fitness_func(candidates)
            if fitness > best_fitness:
                best_fitness = fitness
                best_coordinates = candidates.copy()
```
        ''',
                                           readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences",
                                  enum_options=['max', 'coordinates', 'best_max', 'best_solution'],
                                  readonly=True),
            'time_constraint': Parameter(short_name="time_constraint", type=PrimitiveType.time, long_name='Time limit for evaluation',
                                  description="Time constraint in seconds specifying the maximum runtime of the generated algorithm",
                                  min_value=0, max_value=31536000, default=60)
        }

    def _init_params(self):
        super()._init_params()
        self.time_constraint = self.parameters.get('time_constraint', self.get_parameters().get('time_constraint').default)



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
            best_sol_text = self._best.get_input()
        else:
            best_sol_text = "No best solution yet."

        solution.set_fitness(-1)

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
            coordinates, fitness = algorithm(_evaluate_found_points, _check_inside_triangle, self.time_constraint)
            solution.set_fitness(fitness)

            solution.add_metadata('results', {'coordinates': coordinates, 'fitness': fitness})

            self._check_if_best(solution)

            self._keys = {
                'max': fitness,
                'best_max': self._best.get_metadata().get('results').get('fitness'),
                'coordinates': coordinates,
                'best_solution': best_sol_text
            }

            feedback = self.get_feedback_msg_template().format(**self._keys)
            solution.set_feedback(feedback)

        except Exception as e:
            logging.error('Evaluator:Heilbronn-11: Error during Task evaluation: ' + repr(e))
            raise e

        return fitness

    @classmethod
    def get_short_name(cls) -> str:
        return "eval.heilbronn"

    @classmethod
    def get_long_name(cls) -> str:
        return "Heilbronn-11"

    @classmethod
    def get_description(cls) -> str:
        return "Evaluator for algorithms solving Heilbronn 11. Assuming generated Python code with " \
               "template code of the runner."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'python'},
            'output': {'heilbronn'}
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

        if self._best is None:
            self._best = copy.deepcopy(solution)
            self._keys['best_solution'] = self._best.get_input()
            return True

        if solution.get_fitness() >= self._best.get_fitness():
            self._best = copy.deepcopy(solution)
            self._keys['best_solution'] = self._best.get_input()
            return True
        return False
