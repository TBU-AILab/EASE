from .ps_utils.paintshop_init import PaintshopFunction
from ...modul import Modul


class Paintshop(Modul):

    @classmethod
    def get_short_name(cls) -> str:
        return "resource.paintshop"

    @classmethod
    def get_long_name(cls) -> str:
        return "Paintshop benchmark"

    @classmethod
    def get_description(cls) -> str:
        return "Paintshop benchmark with options:\ntrajectory_x - testing trajectory #x\nfull - all trajectories"

    @staticmethod
    def full() -> list[dict]:
        """
        Returns the whole benchmark specification fit for EvaluatorPaintshopv2
        """
        funcs = [PaintshopFunction(funcNum=f_num) for f_num in range(0, 8)]

        # Prepare test functions
        functions = []
        i = 1
        for f in funcs:
            f_dict = {
                'func': f,
                'dim': f.CF_info(),
                'init': f.CF_extract(),
                'index': i
            }
            functions.append(f_dict)
            i += 1

        return functions

    @staticmethod
    def trajectory1() -> list[dict]:
        """
        Returns the whole benchmark specification fit for EvaluatorPaintshopv2
        """
        funcs = [PaintshopFunction(funcNum=0)]

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'dim': f.CF_info(),
                'init': f.CF_extract(),
                'index': 1
            }
            functions.append(f_dict)

        return functions

    @staticmethod
    def trajectory2() -> list[dict]:
        """
        Returns the whole benchmark specification fit for EvaluatorPaintshopv2
        """
        funcs = [PaintshopFunction(funcNum=1)]

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'dim': f.CF_info(),
                'init': f.CF_extract(),
                'index': 2
            }
            functions.append(f_dict)

        return functions

    @staticmethod
    def trajectory3() -> list[dict]:
        """
        Returns the whole benchmark specification fit for EvaluatorPaintshopv2
        """
        funcs = [PaintshopFunction(funcNum=2)]

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'dim': f.CF_info(),
                'init': f.CF_extract(),
                'index': 3
            }
            functions.append(f_dict)

        return functions

    @staticmethod
    def trajectory4() -> list[dict]:
        """
        Returns the whole benchmark specification fit for EvaluatorPaintshopv2
        """
        funcs = [PaintshopFunction(funcNum=3)]

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'dim': f.CF_info(),
                'init': f.CF_extract(),
                'index': 4
            }
            functions.append(f_dict)

        return functions

    @staticmethod
    def trajectory5() -> list[dict]:
        """
        Returns the whole benchmark specification fit for EvaluatorPaintshopv2
        """
        funcs = [PaintshopFunction(funcNum=4)]

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'dim': f.CF_info(),
                'init': f.CF_extract(),
                'index': 5
            }
            functions.append(f_dict)

        return functions

    @staticmethod
    def trajectory6() -> list[dict]:
        """
        Returns the whole benchmark specification fit for EvaluatorPaintshopv2
        """
        funcs = [PaintshopFunction(funcNum=5)]

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'dim': f.CF_info(),
                'init': f.CF_extract(),
                'index': 6
            }
            functions.append(f_dict)

        return functions

    @staticmethod
    def trajectory7() -> list[dict]:
        """
        Returns the whole benchmark specification fit for EvaluatorPaintshopv2
        """
        funcs = [PaintshopFunction(funcNum=6)]

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'dim': f.CF_info(),
                'init': f.CF_extract(),
                'index': 7
            }
            functions.append(f_dict)

        return functions

    @staticmethod
    def trajectory8() -> list[dict]:
        """
        Returns the whole benchmark specification fit for EvaluatorPaintshopv2
        """
        funcs = [PaintshopFunction(funcNum=7)]

        # Prepare test functions
        functions = []
        for f in funcs:
            f_dict = {
                'func': f,
                'dim': f.CF_info(),
                'init': f.CF_extract(),
                'index': 8
            }
            functions.append(f_dict)

        return functions

