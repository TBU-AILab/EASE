import numpy as np
import logging
from datetime import timedelta, datetime


class MaxEvalException(Exception):

    def __init__(self, a, f):
        self.text = f'Algorithm {a.__module__} tried to exceed maximum number of evaluations on function = {f}.'


class MaxTimeException(Exception):

    def __init__(self, a, f):
        self.text = f'Algorithm {a.__module__} tried to exceed maximum time of evaluations on function = {f}.'

class DimException(Exception):

    def __init__(self, a, dim, length, f):
        self.text = f'Algorithm {a.__module__} passed array of length = {length} for problem of dim = {dim} on function = {f}.'


class Runner():

    def __init__(self, alg, func, dim, bounds, max_evals, max_time):

        self._max_time = timedelta(seconds=max_time)
        self._time_start = datetime.now()
        self._evals = 0
        self._max_evals = max_evals
        self._dim = dim
        self._bounds = bounds
        self._func = func
        # Result
        self._best = {
            'params': np.array([]),
            'fitness': None,
            'eval_num': 0
        }
        self._flag_OoB = False
        self._flag_MaxEval = False
        self._flag_Dim = False
        self._flag_MaxTime = False

        self._a = alg

        pass

    def _func_eval_helper(self, x):

        _time = datetime.now() - self._time_start
        # Max time check
        if _time >= self._max_time:
            self._flag_MaxTime = True
            raise MaxTimeException(self._a, self._func)

        # Max evaluations check
        if self._evals >= self._max_evals:
            if not self._flag_MaxEval:
                self._flag_MaxEval = True
            self._evals += 1
            raise MaxEvalException(self._a, self._func)

        # Check dim vs len(x)
        if self._dim != len(x):
            if not self._flag_Dim:
                self._flag_Dim = True
            raise DimException(self._a, self._dim, len(x), self._func)

        # Out of bounds check
        xx = np.clip(x, self._bounds[:, 0], self._bounds[:, 1])
        comp = x == xx
        equal_arr = comp.all()
        if equal_arr == False:
            if not self._flag_OoB:
                self._flag_OoB = True

        ret = self._func.evaluate(xx)

        # Logging best found value
        if self._best['fitness'] is None or ret <= self._best['fitness']:
            self._best['fitness'] = ret
            self._best['params'] = xx
            self._best['eval_num'] = self._evals

        self._evals += 1

        return ret

    def run(self) -> dict:
        data = {}
        try:
            self._time_start = datetime.now()
            self._a(self._func_eval_helper, self._dim, self._bounds, self._max_evals)
        except MaxEvalException as e:
            logging.warning(f'ResourceTask:Metaheuristic:Runner: {e.text}')
            data['maxevalexception'] = e.text
        except MaxTimeException as e:
            logging.warning(f'ResourceTask:Metaheuristic:Runner: {e.text}')
            data['maxtimeexception'] = e.text
        except DimException as e:
            logging.error(f'ResourceTask:Metaheuristic:Runner: {e.text}')
            data['dimexception'] = e.text
        # for checking general unexpected exceptions
        except Exception as e:
            logging.error(f'ResourceTask:Metaheuristic:Runner: {e}')
            data['unexpectedexception'] = f'Algorithm {self._a.__module__} raised unexpected exception {e} on function = {self._func}.'

        if self._flag_OoB:
            data['outofboundsexception'] = f'ResourceTask:Metaheuristic:Runner: Algorithm {self._a.__module__} tried to evaluate out of bounds. Parameters were clipped to bounds.'
            logging.warning(
                f'ResourceTask:Metaheuristic:Runner: Algorithm {self._a.__module__} tried to evaluate out of bounds. Parameters were clipped to bounds.')
        data['best'] = self._best

        return data
