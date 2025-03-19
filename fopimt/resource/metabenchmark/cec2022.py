from ...modul import Modul
from .CEC2022_data.CEC2022_init import CEC2022function


class CEC2022(Modul):

    @classmethod
    def get_short_name(cls) -> str:
        return "resource.cec2022"

    @classmethod
    def get_long_name(cls) -> str:
        return "CEC 2022 benchmark"

    @classmethod
    def get_description(cls) -> str:
        return "CEC 2022 benchmark (https://github.com/P-N-Suganthan/2022-SO-BO) with two options:\nsample - testing sample\ndim10 - only 10D\ndim20 - only 20D\nfull - whole benchmark"

    @staticmethod
    def whole() -> list[dict]:
        """
        Returns the whole benchmark specification fit for EvaluatorMetaheuristic
        """

        dims = [10, 20]
        funcs = [CEC2022function(funcNum=f_num, dim=dim) for f_num in range(1, 13) for dim in dims]
        runs = 30

        # Prepare test functions
        functions = []

        for f in funcs:
            dim = f.dim
            if dim == 10:
                max_fes = 200_000
            else:
                max_fes = 1_000_000
            f_dict = {
                'func': f,
                'runs': runs,
                'dim': dim,
                'max_fes': max_fes
            }
            functions.append(f_dict)

        return functions

    @staticmethod
    def dim10() -> list[dict]:
        """
        Returns the filtered benchmark specification fit for EvaluatorMetaheuristic
        """

        dims = [10]
        funcs = [CEC2022function(funcNum=f_num, dim=dim) for f_num in range(1, 13) for dim in dims]
        runs = 30

        # Prepare test functions
        functions = []

        for f in funcs:
            dim = f.dim
            if dim == 10:
                max_fes = 200_000
            else:
                max_fes = 1_000_000
            f_dict = {
                'func': f,
                'runs': runs,
                'dim': dim,
                'max_fes': max_fes
            }
            functions.append(f_dict)

        return functions

    @staticmethod
    def dim20() -> list[dict]:
        """
        Returns the filtered benchmark specification fit for EvaluatorMetaheuristic
        """

        dims = [20]
        funcs = [CEC2022function(funcNum=f_num, dim=dim) for f_num in range(1, 13) for dim in dims]
        runs = 30

        # Prepare test functions
        functions = []

        for f in funcs:
            dim = f.dim
            if dim == 10:
                max_fes = 200_000
            else:
                max_fes = 1_000_000
            f_dict = {
                'func': f,
                'runs': runs,
                'dim': dim,
                'max_fes': max_fes
            }
            functions.append(f_dict)

        return functions

    @staticmethod
    def sample() -> list[dict]:
        """
        Returns a small sample of the benchmark for testing purposes, fit for EvaluatorMetaheuristic
        """
        funcs = [CEC2022function(funcNum=f_num, dim=10) for f_num in range(1, 3)]
        max_fes = 1_000
        runs = 3

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'runs': runs,
                'dim': f.dim,
                'max_fes': max_fes
            }
            functions.append(f_dict)
        return functions
