import copy

from .evaluator import Evaluator
from ..solutions.solution import Solution


class EvaluatorDummy(Evaluator):

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate(self, solution: Solution) -> float:
        """
        Evaluation function. Returns quality of solution as float number.
        Arguments:
            solution: Solution  -- Solution that will be evaluated.
        """
        # Evaluation of the fitness and fitness setting
        if self._best is not None:
            fitness = self._best.get_fitness() + 1
        else:
            fitness = 0
        solution.set_fitness(fitness)
        solution.add_metadata('OK', True)

        self._check_if_best(solution)

        return fitness


    @classmethod
    def get_short_name(cls) -> str:
        return "eval.dummy"

    @classmethod
    def get_long_name(cls) -> str:
        return "Dummy"

    @classmethod
    def get_description(cls) -> str:
        return "Dummy (simple) evaluator. Threat each evaluation of the solution as the better one."

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
