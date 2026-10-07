import numpy as np

from ...modul import Modul


class BBOBfunction:
    """
    Noiseless BBOB function from the official COCO implementation. cocoex.BareProblem is used because, unlike
    cocoex.Suite, it allows any dimension (e.g. D=30). evaluate() returns the error f(x) - f(x*).
    The search domain is the bbob region of interest [-5, 5]^D.
    The COCO problem cannot be pickled or deep-copied, it is therefore created lazily and dropped from the state.
    """

    def __init__(self, funcNum: int, dim: int, instance: int = 1):
        self.func = funcNum
        self.dim = dim
        self.instance = instance
        self._problem = None
        self.optimum_value = self._get_problem().best_value()

    def _get_problem(self):
        if self._problem is None:
            import cocoex

            self._problem = cocoex.BareProblem(
                "bbob", self.func, self.dim, self.instance
            )
        return self._problem

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_problem"] = None
        return state

    def get_bounds(self):
        return np.array([[-5.0, 5.0]] * self.dim)

    def evaluate(self, x) -> float:
        return self._get_problem()(np.asarray(x, dtype=float)) - self.optimum_value

    def __str__(self) -> str:
        return f"bbob_f{self.func:03d}_i{self.instance:02d}_d{self.dim}"


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
            "COCO BBOB noiseless benchmark (https://coco-platform.org/testsuites/bbob/overview.html), official COCO "
            "implementation (cocoex). Options:\n"
            "f_24 - Lunacek bi-Rastrigin, instance 1, D=30"
        )

    @staticmethod
    def f_24() -> dict:
        """
        Returns f24 (Lunacek bi-Rastrigin), instance 1, D=30
        """
        func = BBOBfunction(funcNum=24, dim=30, instance=1)
        return {"func": func, "runs": 30, "dim": func.dim, "max_fes": 1_000_000}
