# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 17:12:49 2023

@author: kadavy
"""
import random
import math

import numpy

from .TaskC_FormulaNewPoints import evaluate
from .TaskC_FormulaNewPoints import wx

soft_cons = True

#limits (bounds) of each parameter
# lims = numpy.array([
#         [0, 10],       #b0D
#         [0, 100],       #tau0
#         [0, 250],      #tau
#         [0, 1000],      #a2
#         [0, 1000],      #a1
#         [-100, 100],    #a0
#         [-100, 100],    #a0D
#         [0, 250]       #theta
#         ])

# lims = numpy.array([
#         [0, 100],       #b0D
#         [0, 500],       #tau0
#         [0, 1000],      #tau
#         [0, 2500],      #a2
#         [0, 2500],      #a1
#         [-250, 250],    #a0
#         [-250, 250],    #a0D
#         [0, 1000]       #theta
#         ])
#
lims = numpy.array([
        [0, 500],       #b0D
        [0, 500],       #tau0
        [0, 1000],      #tau
        [0, 2500],      #a2
        [0, 2500],      #a1
        [-250, 250],    #a0
        [-250, 250],    #a0D
        [0, 1000]       #theta
        ])

# copy for Tom approach
lims0 = numpy.array([
        [0, 10],       #b0D
        [0, 100],       #tau0
        [0, 250],      #tau
        [0, 1000],      #a2
        [0, 1000],      #a1
        [-100, 100],    #a0
        [-100, 100],    #a0D
        [0, 250]       #theta
        ])

lims1 = numpy.array([
        [0, 100],       #b0D
        [0, 500],       #tau0
        [0, 1000],      #tau
        [0, 2500],      #a2
        [0, 2500],      #a1
        [-250, 250],    #a0
        [-250, 250],    #a0D
        [0, 1000]       #theta
        ])

lims2 = numpy.array([
        [0, 500],       #b0D
        [0, 500],       #tau0
        [0, 1000],      #tau
        [0, 2500],      #a2
        [0, 2500],      #a1
        [-250, 250],    #a0
        [-250, 250],    #a0D
        [0, 1000]       #theta
        ])

lims = lims1

def checkBorders(x):
    # bound constraints
    for i in range(len(lims)):
        if (lims[i][0] <= x[i] <= lims[i][1]) == False:
            return False

def checkConstraints(x):
    #constraints
    if x[1] <= 0:
        return False
    if x[2] <= 0:
        return False
    if x[7] <= 0:
        return False
    if x[3] <= 0:
        return False
    if x[4] <= 0:
        return False
    if x[5] + x[6] <= 0:
        return False
    if x[3] * x[4] <= x[5]:
        return False
    if x[3] * x[4] <= x[5] + x[6]:
        return False
    if x[0] == 0:
        return False
    if x[5] == 0:
        return False
    if x[6] == 0:
        return False
    for ii in wx:
        omega = ii**2
        if x[6] / math.sqrt((x[5]- x[3] * omega)**2 + omega * (x[4] - omega)**2) >= 1:
            return False
    return True

def checkFeasibility(x):
    if (checkBorders(x) == False) :
        return False
    if checkConstraints(x) == False :
        return False
    return True


def generateFeasibleIndividual():
    x = [random.uniform(lims[j][0], lims[j][1]) for j in range(len(lims))]
    while checkFeasibility(x) != True:
        #print("ted")
        x = [random.uniform(lims[j][0], lims[j][1]) for j in range(len(lims))]
    return x    


def generateInBoundINdividual():
    x = [random.uniform(lims[j][0], lims[j][1]) for j in range(len(lims))]
    return x




#y = [6.09016918e-03,1.40180516e+01,1.35433245e+02,8.87416231e+02,6.18164390e+01,1.10477816e+00,-5.94615425e-01,1.50996895e+02]


def CostFunctionSingleSolution(x):
    return evaluate(x)


def CostFunction(x):
    m, n = numpy.shape(x)
    f = numpy.zeros(n)

    for j in range(0, n):
        f[j] = CostFunctionSingleSolution(x[:, j])
        '''
        if checkFeasibility(x[:,j]) == False:
            f[j] = 10**5
        else:
            f[j] = CostFunctionSingleSolution(x[:,j])

        '''
    # print(f[j])
    return f


# tomova ohodnocujici funkce
# vraci vzdy array [f, feasible]
class VsePohlcujiciUzasnaFunkce:
    def __init__(self):
        self._resulto = 0

    def get_bounds(self):
        return lims

    def __str__(self) -> str:
        return str(1)

    def evaluate(self, x_in):
        x = x_in.tolist()
        #print("WTF: "+str(x))
        reseni = self._resulto
        if reseni == 0:
            # HARD - C
            if checkFeasibility(x) == False:
                x = generateFeasibleIndividual()
            _f = evaluate(x)
            return _f
        if reseni == 1:
            # SOFT - C
            if checkBorders(x) == False:
                x = generateInBoundINdividual()
            _f = evaluate(x)
            _cTest = checkConstraints(x)
            if _cTest == False:
                _f += 10 ** 5
            return _f
        if reseni == 2:
            # NO-C
            if checkBorders(x) == False:
                x = generateInBoundINdividual()
            return evaluate(x)

            # jen pro jistotu
        return 10 ** 5


class PolskoGACR:
    def __init__(self):
        pass

    def get_whole_benchmark(self) -> list[dict]:
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

    def get_small_sample(self) -> list[dict]:
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
    





