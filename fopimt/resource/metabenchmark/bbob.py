import numpy as np

from ...modul import Modul


class BBOBfunction:
    """
    Noiseless BBOB function (COCO) via IOHexperimenter, which reproduces the COCO bbob suite and, unlike COCO,
    supports any dimension. evaluate() returns the error f(x) - f(x*).
    The ioh problem cannot be pickled or deep-copied, it is therefore created lazily and dropped from the state.
    """

    def __init__(self, funcNum: int, dim: int, instance: int = 1):
        self.func = funcNum
        self.dim = dim
        self.instance = instance
        self._problem = None
        self.optimum_value = self._get_problem().optimum.y

    def _get_problem(self):
        if self._problem is None:
            import ioh

            self._problem = ioh.get_problem(
                self.func,
                instance=self.instance,
                dimension=self.dim,
                problem_class=ioh.ProblemClass.BBOB,
            )
        return self._problem

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_problem"] = None
        return state

    def get_bounds(self):
        problem = self._get_problem()
        return np.array([problem.bounds.lb, problem.bounds.ub]).T

    def evaluate(self, x) -> float:
        return self._get_problem()(np.asarray(x, dtype=float)) - self.optimum_value

    def __str__(self) -> str:
        return f"BBOB-f{self.func}-i{self.instance}-D{self.dim}"


class BBOB(Modul):
    @classmethod
    def get_short_name(cls) -> str:
        return "resource.bbob"

    @classmethod
    def get_long_name(cls) -> str:
        return "BBOB benchmark"

    @classmethod
    def get_description(cls) -> str:
        return (
            "COCO BBOB noiseless benchmark (https://coco-platform.org/testsuites/bbob/overview.html) via IOHexperimenter. "
            "Options:\n"
            "f_24 - Lunacek bi-Rastrigin, instance 1, D=30"
        )

    @staticmethod
    def f_24() -> dict:
        """
        Returns f24 (Lunacek bi-Rastrigin), instance 1, D=30
        """
        func = BBOBfunction(funcNum=24, dim=30, instance=1)
        return {"func": func, "runs": 30, "dim": func.dim, "max_fes": 1_000_000}
