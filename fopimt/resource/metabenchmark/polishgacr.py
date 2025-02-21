from ...modul import Modul
from .polskoGACR_data.fit import VsePohlcujiciUzasnaFunkce


class PolishGACR(Modul):

    @classmethod
    def get_short_name(cls) -> str:
        return "resource.polishgacr"

    @classmethod
    def get_long_name(cls) -> str:
        return "Benchmark for Polish GACR"

    @classmethod
    def get_description(cls) -> str:
        return "Benchmark for Polish GACR with two options:\nsample - testing sample\nfull - whole benchmark"

    @staticmethod
    def whole() -> list[dict]:
        """
        Returns a small sample of the benchmark for testing purposes, fit for EvaluatorMetaheuristic
        """
        funcs = [VsePohlcujiciUzasnaFunkce()]
        runs = 31

        # Let's set our Evaluator type.
        # functions = [{
        #     'func': VsePohlcujiciUzasnaFunkce,
        #     'runs': 11,
        #     'dim': lims,
        #     'max_fes': 500_000
        # }]

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'runs': runs,
                'dim': 8,
                'max_fes': 500_000
            }
            functions.append(f_dict)
        return functions

    @staticmethod
    def sample() -> list[dict]:
        """
        Returns a small sample of the benchmark for testing purposes, fit for EvaluatorMetaheuristic
        """
        funcs = [VsePohlcujiciUzasnaFunkce()]
        runs = 2

        # Let's set our Evaluator type.
        # functions = [{
        #     'func': VsePohlcujiciUzasnaFunkce,
        #     'runs': 11,
        #     'dim': lims,
        #     'max_fes': 500_000
        # }]

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'runs': runs,
                'dim': 8,
                'max_fes': 5_000
            }
            functions.append(f_dict)
        return functions