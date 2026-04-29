import copy

from fopimt.task_dto import OptimizationGoal

from ..solutions.solution import Solution
from .evaluator import Evaluator, EvaluatorResult


class EvaluatorDummy(Evaluator):
    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate(
        self,
        solution: Solution,
        opt_goal: OptimizationGoal = OptimizationGoal.MINIMIZATION,
    ) -> EvaluatorResult:
        """
        Evaluation function. Returns quality of solution as EvaluatorResult.
        Arguments:
            solution: Solution  -- Solution that will be evaluated.
        """
        # Evaluation of the fitness and fitness setting
        if self._best is not None:
            fitness = self._best.get_fitness() + 1
        else:
            fitness = 0
        solution.set_fitness(fitness)
        solution.add_metadata("OK", True)

        self._check_if_best(solution)

        result_metadata = {"OK": True}
        return EvaluatorResult(
            class_ref=type(self), fitness=fitness, metadata=result_metadata
        )

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
        return {"input": set(), "output": set()}

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
