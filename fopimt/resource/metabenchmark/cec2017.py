from ...modul import Modul
from .CEC2017_data.CEC2017_init import CEC2017function


class CEC2017(Modul):
    @classmethod
    def get_short_name(cls) -> str:
        return "resource.cec2017"

    @classmethod
    def get_long_name(cls) -> str:
        return "CEC 2017 benchmark"

    @classmethod
    def get_description(cls) -> str:
        return (
            "CEC 2017 single objective bound constrained benchmark (https://github.com/P-N-Suganthan/CEC2017-BoundContrained), "
            "test suite of the CEC 2024 and CEC 2026 competitions. Only D=30 data are bundled. Options:\n"
            "dim30 - 29 functions (F2 excluded as in the competitions), D=30\n"
            "f_30 - Composition Function 10 (F30), D=30"
        )

    @staticmethod
    def dim30() -> list[dict]:
        """
        Returns the competition setting (29 functions without F2, D=30, Max_FEs = 10000*D), fit for EvaluatorMetaheuristic
        """
        dim = 30
        return [
            {
                "func": CEC2017function(funcNum=f_num, dim=dim),
                "runs": 30,
                "dim": dim,
                "max_fes": 10_000 * dim,
            }
            for f_num in range(1, 31)
            if f_num != 2
        ]

    @staticmethod
    def f_30() -> dict:
        """
        Returns F30 (Composition Function 10), D=30
        """
        func = CEC2017function(funcNum=30, dim=30)
        return {"func": func, "runs": 30, "dim": func.dim, "max_fes": 1_000_000}
