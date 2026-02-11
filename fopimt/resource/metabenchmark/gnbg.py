from ...modul import Modul
from .gnbg_data.gnbg_init import GNBGfunction


class GNBG(Modul):

    @classmethod
    def get_short_name(cls) -> str:
        return "resource.gnbg"

    @classmethod
    def get_long_name(cls) -> str:
        return "GNBG benchmark"

    @classmethod
    def get_description(cls) -> str:
        return "GNBG benchmark (https://github.com/Danial-Yazdani/GNBG_Instances.Python) with two options:\nsample - testing sample\nfull - whole benchmark"

    @staticmethod
    def full() -> list[dict]:
        """
        Returns the whole benchmark specification fit for EvaluatorMetaheuristic
        """
        funcs = [GNBGfunction(funcNum=f_num) for f_num in range(1, 25)]
        runs = 31

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'runs': runs,
                'dim': f.dim,
                'max_fes': f.maxfes
            }
            functions.append(f_dict)

        return functions

    @staticmethod
    def sample() -> list[dict]:
        """
        Returns a small sample of the benchmark for testing purposes, fit for EvaluatorMetaheuristic
        """
        funcs = [GNBGfunction(funcNum=f_num) for f_num in range(1, 3)]
        runs = 8

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'runs': runs,
                'dim': f.dim,
                'max_fes': 1_000
            }
            functions.append(f_dict)
        return functions

    @staticmethod
    def f_24() -> dict:
        """
        Returns a func n 24
        """

        func = GNBGfunction(funcNum=24)
        runs = 30

        # Prepare test functions
        f_dict = {
            'func': func,
            'runs': runs,
            'dim': func.dim,
            'max_fes': 1_000_000
        }
        return f_dict
