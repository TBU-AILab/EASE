import numpy as np
from . import GNBG_instances
import os
from scipy.io import loadmat


class GNBGfunction:

    def __init__(self, funcNum: int):
        self._ProblemIndex = funcNum

        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the path to the folder where you want to read/write files
        folder_path = os.path.join(current_dir)
        np.random.seed()

        filename = f'f{self._ProblemIndex}.mat'
        GNBG_tmp = loadmat(os.path.join(folder_path, filename))['GNBG']
        MaxEvals = np.array([item[0] for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
        AcceptanceThreshold = np.array([item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
        Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
        CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]  # Number of components
        MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
        MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
        CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
        CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
        CompH = np.array(GNBG_tmp['Component_H'][0, 0])
        Mu = np.array(GNBG_tmp['Mu'][0, 0])
        Omega = np.array(GNBG_tmp['Omega'][0, 0])
        Lambda = np.array(GNBG_tmp['lambda'][0, 0])
        RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
        OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
        OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])

        gnbg = GNBG_instances.GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate,
                                   CompMinPos, CompSigma,
                                   CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)

        self.dim = Dimension
        self.maxfes = MaxEvals
        self._gnbg = gnbg
        self._bounds = np.array([[gnbg.MinCoordinate, gnbg.MaxCoordinate]] * self.dim)

    def evaluate(self, x):
        X = np.array([x])
        return abs(self._gnbg.fitness(X)[0] - self._gnbg.OptimumValue)

    def get_bounds(self):
        return self._bounds

    def __str__(self) -> str:
        return str(self._ProblemIndex)


