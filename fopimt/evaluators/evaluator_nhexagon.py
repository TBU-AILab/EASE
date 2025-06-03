import logging
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
from .evaluator import Evaluator
from ..solutions.solution import Solution
from ..loader import Parameter, PrimitiveType
import copy
from ..utils.import_utils import dynamic_import
from ..resource.resource import Resource, ResourceType
import sys

    
def hexagon_vertices(
    center_x: float,
    center_y: float,
    side_length: float,
    angle_degrees: float,
) -> list[tuple[float, float]]:
    """Generates the vertices of a regular hexagon.

    Args:
    center_x: x-coordinate of the center.
    center_y: y-coordinate of the center.
    side_length: Length of each side.
    angle_degrees: Rotation angle in degrees (clockwise from horizontal).

    Returns:
    A list of tuples, where each tuple (x, y) represents the vertex location.
    """
    vertices = []
    angle_radians = math.radians(angle_degrees)
    for i in range(6):
        angle = angle_radians + 2 * math.pi * i / 6
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices


def normalize_vector(v: tuple[float, float]) -> tuple[float, float]:
    """Normalizes a 2D vector."""
    magnitude = math.sqrt(v[0]**2 + v[1]**2)
    return (v[0] / magnitude, v[1] / magnitude) if magnitude != 0 else (0., 0.)


def get_normals(
    vertices: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Gets the outward normals of a polygon's edges."""
    normals = []
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]  # Wrap around to the first vertex.
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = normalize_vector((-edge[1], edge[0]))  # Rotate edge by 90 degrees.
        normals.append(normal)
    return normals


def project_polygon(
    vertices: list[tuple[float, float]],
    axis: tuple[float, float],
) -> tuple[float, float]:
    """Projects a polygon onto an axis and returns the min/max values."""
    min_proj = float('inf')
    max_proj = float('-inf')
    for vertex in vertices:
        projection = vertex[0] * axis[0] + vertex[1] * axis[1]  # Dot product.
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    return min_proj, max_proj


def overlap_1d(min1: float, max1: float, min2: float, max2: float) -> bool:
    """Determines whether two 1D intervals overlap."""
    return max1 >= min2 and max2 >= min1


def polygons_intersect(
    vertices1: list[tuple[float, float]],
    vertices2: list[tuple[float, float]],
) -> bool:
    """Determines if two polygons intersect using the Separating Axis Theorem."""
    normals1 = get_normals(vertices1)
    normals2 = get_normals(vertices2)
    axes = normals1 + normals2

    for axis in axes:
        min1, max1 = project_polygon(vertices1, axis)
        min2, max2 = project_polygon(vertices2, axis)
        if not overlap_1d(min1, max1, min2, max2):
            return False  # Separating axis found, polygons are disjoint.
    return True  # No separating axis found, polygons intersect.


def hexagons_are_disjoint(
    hex1_params: tuple[float, float, float, float],
    hex2_params: tuple[float, float, float, float],
) -> bool:
    """Determines if two hexagons are disjoint given their parameters."""
    hex1_vertices = hexagon_vertices(*hex1_params)
    hex2_vertices = hexagon_vertices(*hex2_params)
    return not polygons_intersect(hex1_vertices, hex2_vertices)


def is_inside_hexagon(
    point: tuple[float, float],
    hex_params: tuple[float, float, float, float],
) -> bool:
    """Checks if a point is inside a hexagon (given its parameters)."""
    hex_vertices = hexagon_vertices(*hex_params)
    for i in range(len(hex_vertices)):
        p1 = hex_vertices[i]
        p2 = hex_vertices[(i + 1) % len(hex_vertices)]
        edge_vector = (p2[0] - p1[0], p2[1] - p1[1])
        point_vector = (point[0] - p1[0], point[1] - p1[1])
        cross_product = (
        edge_vector[0] * point_vector[1] - edge_vector[1] * point_vector[0]
        )
        if cross_product < 0:
            return False
    return True


def all_hexagons_contained(
    inner_hex_params_list: list[tuple[float, float, float, float]],
    outer_hex_params: tuple[float, float, float, float],
) -> bool:
    """Checks if all inner hexagons are contained within the outer hexagon."""
    for inner_hex_params in inner_hex_params_list:
        inner_hex_vertices = hexagon_vertices(*inner_hex_params)
        for vertex in inner_hex_vertices:
            if not is_inside_hexagon(vertex, outer_hex_params):
                return False
    return True


def verify_construction(
    inner_hex_data: tuple[float, float, float],
    outer_hex_center: tuple[float, float],
    outer_hex_side_length: float,
    outer_hex_angle_degrees: float,
):
    """Verifies the hexagon packing construction with a rotated outer hexagon.

    Args:
    inner_hex_data: List of (x, y, angle_degrees) tuples for inner hexagons.
    outer_hex_center: (x, y) tuple for the outer hexagon center.
    outer_hex_side_length: Side length of the outer hexagon.
    outer_hex_angle_degrees: Rotation angle of the outer hexagon in degrees.

    Raises:
    AssertionError if the construction is not valid.
    """

    inner_hex_params_list = [
      (x, y, 1, angle) for x, y, angle in inner_hex_data
    ]  # Sets the side length to 1.
    outer_hex_params = (
      outer_hex_center[0], outer_hex_center[1],
      outer_hex_side_length, outer_hex_angle_degrees
    )

    # Disjointness check.
    for i in range(len(inner_hex_params_list)):
        for j in range(i + 1, len(inner_hex_params_list)):
            if not hexagons_are_disjoint(
              inner_hex_params_list[i], inner_hex_params_list[j]
            ):
                return False

    # Containment check.
    if not all_hexagons_contained(inner_hex_params_list, outer_hex_params):
        return False

    return True


class EvaluatorHexagon(Evaluator):
    '''
    Evaluator for Hexagon algorithms. The algorithm must be Python code and follow predefined structure.

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
                                               default="Is the found result viable? {viable}\nThe result found by the current solution is: {outer_hex_side_length}\nBest-so-far solution: {best_solution}\nand the best result is: {best_outer_hex_side_length}\n"
                                               ),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.markdown,
                                           long_name="Template for an initial message",
                                           description="Initial message for evaluation. Specific for each evaluator.",
                                           default='''
Your task is to design an efficient algorithm to solve the *unit regular hexagon packing problem*.

The objective is to pack `n` *unit regular hexagons* (side length = 1) inside a single *rotated regular hexagon* such that:

* The packed inner hexagons are **disjoint**.
* All inner hexagons are **fully contained** within the outer hexagon.
* The side length of the outer hexagon is **as small as possible**.

You must return a configuration that passes the provided `verify_construction` function. Your algorithm should:

* Be implemented within the following Python function:

  ```
  def algorithm(n, verify_construction, time_constraint):
      # Your implementation here
      return inner_hex_data, outer_hex_center, outer_hex_side_length, outer_hex_angle_degrees
  ```
* Accept:

  * `n`: number of inner hexagons to place
  * `verify_construction`: function that checks validity of your configuration
  * `time_constraint`: maximum time (in seconds) for your algorithm to run
* Return:

  * `inner_hex_data`: list of `(x, y, angle_degrees)` for each inner hexagon
  * `outer_hex_center`: `(x, y)` of the outer hexagon’s center
  * `outer_hex_side_length`: side length (float) of the outer hexagon
  * `outer_hex_angle_degrees`: rotation angle (float) of the outer hexagon

You may use any suitable method, including grid placement, ring-based construction, greedy shrinking, or symmetry-aware strategies. Metaheuristics are permitted but not required. Be creative but adhere strictly to time and geometric constraints.

Avoid any form of testing or logging. Only return the best valid configuration you find.

''',
                                           readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences",
                                  enum_options=['best_inner_hex_data',
                'best_outer_hex_center',
                'best_outer_hex_side_length',
                "best_outer_hex_angle_degrees",
                'inner_hex_data',
                'outer_hex_center',
                'outer_hex_side_length',
                'outer_hex_angle_degrees',
                'best_solution',
                'viable'],
                                  readonly=True),
            'time_constraint': Parameter(short_name="time_constraint", type=PrimitiveType.time, long_name='Time limit for evaluation',
                                  description="Time constraint in seconds specifying the maximum runtime of the generated algorithm",
                                  min_value=0, max_value=31536000, default=60),
            'point_count': Parameter(short_name="point_count", type=PrimitiveType.int,
                                         long_name='Number of points to place (n)',
                                         description="How many points should be placed inside of the triangle - parameter n.",
                                         min_value=11, max_value=12, default=12, required=True)
        }

    def _init_params(self):
        super()._init_params()
        self.time_constraint = self.parameters.get('time_constraint', self.get_parameters().get('time_constraint').default)
        self.n = self.parameters.get('point_count',self.get_parameters().get('point_count').default)



    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate(self, solution: Solution) -> float:
        """
        Evaluation function. Returns quality of solution as float number.
        Arguments:
            solution: Solution  -- Solution that will be evaluated.
        """

        solution.set_fitness(sys.float_info.max)
        outer_hex_side_length = sys.float_info.max

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

            algorithm = combined_scope['algorithm']
            inner_hex_data, outer_hex_center, outer_hex_side_length, outer_hex_angle_degrees = algorithm(self.n, verify_construction, self.time_constraint)
            if verify_construction(inner_hex_data, outer_hex_center, outer_hex_side_length, outer_hex_angle_degrees):
                viable = True
                fitness = outer_hex_side_length
            else:
                viable = False
                fitness = sys.float_info.max
            solution.set_fitness(fitness)

            solution.add_metadata('results', {'inner_hex_data': inner_hex_data, 'outer_hex_center': outer_hex_center, "outer_hex_side_length": outer_hex_side_length, "outer_hex_angle_degrees": outer_hex_angle_degrees, 'viable': viable})

            self._check_if_best(solution)

            self._keys = {
                
                'best_inner_hex_data': self._best.get_metadata().get('results').get('inner_hex_data'),
                'best_outer_hex_center': self._best.get_metadata().get('results').get('outer_hex_center'),
                "best_outer_hex_side_length": self._best.get_metadata().get('results').get('outer_hex_side_length'),
                "best_outer_hex_angle_degrees": self._best.get_metadata().get('results').get('outer_hex_angle_degrees'),
                'inner_hex_data': solution.get_metadata().get('results').get('inner_hex_data'),
                'outer_hex_center': solution.get_metadata().get('results').get('outer_hex_center'),
                "outer_hex_side_length": solution.get_metadata().get('results').get('outer_hex_side_length'),
                "outer_hex_angle_degrees": solution.get_metadata().get('results').get('outer_hex_angle_degrees'),
                'best_solution': self._best.get_input(),
                'viable': viable
            }

            feedback = self.get_feedback_msg_template().format(**self._keys)
            solution.set_feedback(feedback)

        except Exception as e:
            logging.error('Evaluator:Hexagon: Error during Task evaluation: ' + repr(e))
            raise e

        return outer_hex_side_length

    @classmethod
    def get_short_name(cls) -> str:
        return "eval.Hexagon"

    @classmethod
    def get_long_name(cls) -> str:
        return "Hexagon"

    @classmethod
    def get_description(cls) -> str:
        return "Evaluator for algorithms solving Hexagon. Assuming generated Python code with " \
               "template code of the runner."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'python'},
            'output': {'Hexagon'}
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
            return True

        if solution.get_fitness() <= self._best.get_fitness():
            self._best = copy.deepcopy(solution)
            return True
        return False