import time
import logging

from .cost import CF_info, CF



class MaxEvalException(Exception):

    def __init__(self, a):
        self.text = f'Algorithm {a.__module__} tried to exceed allowed eval budget.'


class Runner():

    def __init__(self, alg, func, dim, min_speed, max_speed, min_distance, max_distance, max_evals, x_init, y_init):

        self._min_distance = min_distance
        self._max_distance = max_distance
        self._min_speed = min_speed
        self._max_speed = max_speed
        self._max_eval = max_evals
        self._eval = 0
        self._dim = dim
        self._func = func
        self._x_init = x_init
        self._y_init = y_init
        # Result
        self._x_best = list()
        self._y_best = list()
        self._flag_MaxEval = False
        self._a = alg
        self._best_cf = None

    def _func_eval_helper(self, x, y):

        #Max evaluations check
        if self._eval >= self._max_eval:
            if not self._flag_MaxEval:
                self._flag_MaxEval = True
            raise MaxEvalException(self._a)

        self._eval += 1
        fitness = self._func.evaluate(x, y)

        if self._best_cf is None or fitness <= self._best_cf:
            self._best_cf = fitness
            self._x_best = x
            self._y_best = y

        return fitness

    def run(self) -> (float, list, list):
        try:
            self._a(self._func_eval_helper, self._dim, self._min_speed, self._max_speed, self._min_distance, self._max_distance, self._max_eval, self._x_init, self._y_init)
        except MaxEvalException as e:
            logging.warning(f'ResourceTask:Paintshop:Runner_v2: {e.text}')
            return self._best_cf, self._x_best, self._y_best

        return self._best_cf, self._x_best, self._y_best


