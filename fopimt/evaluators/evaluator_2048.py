import logging
import copy
import numpy as np

from ..resource.resource import Resource, ResourceType
from ..resource.game2048.implementation_2048 import new_game, play_2048
from .evaluator import Evaluator
from ..modul import Modul
from ..solutions.solution import Solution
from ..utils.import_utils import dynamic_import
from ..loader import Parameter, PrimitiveType


def play_the_game(move) -> dict:
    returndict = {}

    grid, score = new_game()
    i = 0
    invalid_move_counter = 0
    while True:
        direction = move(grid, score)
        i += 1
        try:
            orig_grid = copy.deepcopy(grid)
            grid, score = play_2048(grid, direction, score)

            if (orig_grid == grid).all():
                invalid_move_counter += 1
            else:
                invalid_move_counter = 0

            if invalid_move_counter >= 20:
                returndict['outcome'] = -1
                break

        except RuntimeError as inst:
            if str(inst) == "GO":
                returndict['outcome'] = 0
            elif str(inst) == "WIN":
                returndict['outcome'] = 1
            break

        except ValueError:
            returndict['outcome'] = -1
            break

    returndict['score'] = score
    returndict['moves'] = i
    returndict['max_tile'] = np.max(grid)

    return returndict


class Evaluator2048(Evaluator):
    """
    Evaluator for 2048 solvers. The algorithm must be Python code and follow predefined structure
    defined in TODO.
    :param params: Various parameters for the algorithm.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            'num_games': Parameter(short_name="num_games", type=PrimitiveType.int,
                                   long_name="Number of games to evaluate solver",
                                   description="Specifies a number of 2048 games that will be played and "
                                               "used to derive the statistics.",
                                   default=5
                                   ),
            'feedback_msg_template': Parameter(short_name="feedback_msg_template", type=PrimitiveType.str,
                                               long_name="Template for a feedback message",
                                               description="Feedback message for evaluation. Can use {keywords}",
                                               default="The metrics of the proposed solver are:\n{wins}{loses}{"
                                                       "errors}{avg_score}{max_score}{min_score}{avg_max_tile}{"
                                                       "avg_steps}"
                                               ),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.str,
                                           long_name="Template for an initial message",
                                           description="Initial message for evaluation. Specific for each evaluator.",
                                           default="""
Implement a Python function move(grid: np.array, score: int) -> 
str to solve the 2048 game. The function should analyze the current 4x4 
game state (with zeros representing empty tiles) and output a move from 
the set {'left', 'right', 'up', 'down'}. Ensure that you adhere strictly 
to the provided structure: method name and attributes are given. Focus on 
developing the [Algorithm body] of the "move" method:

import numpy as np

def move(grid: np.array, score: int) -> str:

    #Function that determines which direction to move the 2048 grid
    #Output: direction - one of possible moves 'left', 'right', 'up' or 'down'

    [Algorithm body]

    return direction

Do not include placeholder text or pseudocode; provide a 
fully functional implementation.""",
                                           readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences",
                                  enum_options=['avg_score', 'max_score', 'min_score', 'avg_max_tile', 'avg_steps',
                                                'wins', 'loses', 'errors'], readonly=True),

        }

    def _init_params(self):
        super()._init_params()
        self._games = self.parameters.get('num_games', 5)

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
                        logging.error("Eval:2048:", repr(e))

            move = combined_scope['move']

            stats = {
                'scores': [],
                'max_tiles': [],
                'steps': [],
                'wins': 0,
                'loses': 0,
                'errors': 0
            }

            for i in range(self._games):
                result = play_the_game(move)

                logging.info(f"Eval:2048:{i + 1}. result: {result}")

                stats['scores'].append(result['score'])
                stats['max_tiles'].append(result['max_tile'])
                stats['steps'].append(result['moves'])

                match result['outcome']:
                    case -1:
                        stats['errors'] += 1
                    case 0:
                        stats['loses'] += 1
                    case 1:
                        stats['wins'] += 1

            results = {
                'avg_score': np.average(stats['scores']),
                'max_score': np.max(stats['scores']),
                'min_score': np.min(stats['scores']),
                'avg_max_tile': np.average(stats['max_tiles']),
                'avg_steps': np.average(stats['steps']),
                'wins': stats['wins'],
                'loses': stats['loses'],
                'errors': stats['errors']
            }

            avg_score_txt = f"The average score of the solver is = {results['avg_score']}\n"
            max_score_txt = f"The maximum score of the solver is = {results['max_score']}\n"
            min_score_txt = f"The minimum score of the solver is = {results['min_score']}\n"
            avg_tile_txt = f"The average maximum tile value of the solver is = {results['avg_max_tile']}\n"
            avg_steps_txt = f"The average move count of the solver is = {results['avg_steps']}\n"
            wins_txt = f"The number of wins of the solver is = {results['wins']}\n"
            loses_txt = f"The number of loses of the solver is = {results['loses']}\n"
            errors_txt = f"The number of times when the solver was stuck or returned invalid move = {results['errors']}\n"

            solution.add_metadata('results', results)

            self._keys = {
                'avg_score': avg_score_txt,
                'max_score': max_score_txt,
                'min_score': min_score_txt,
                'avg_max_tile': avg_tile_txt,
                'avg_steps': avg_steps_txt,
                'wins': wins_txt,
                'loses': loses_txt,
                'errors': errors_txt
            }

            feedback = self.get_feedback_msg_template().format(**self._keys)
            solution.set_feedback(feedback)

            solution.set_fitness(results['avg_score'])
            fitness = results['avg_score']

            self._check_if_best(solution)

        except Exception as e:
            logging.error('Evaluator:Classification: Error during Task evaluation: ' + repr(e))
            raise e

        return fitness

    @classmethod
    def get_short_name(cls) -> str:
        return "eval.2048"

    @classmethod
    def get_long_name(cls) -> str:
        return "2048"

    @classmethod
    def get_description(cls) -> str:
        return "Evaluator for 2048 solvers."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'python'},
            'output': {'solver2048'}
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
