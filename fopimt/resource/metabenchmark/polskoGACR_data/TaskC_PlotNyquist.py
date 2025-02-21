import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sympy import *
from sympy.utilities.lambdify import lambdify


def plot_nyquist_comparison(obtained_values):
    # Nyquist plots (i.e., frequency-domain) comparison
    # Heating model from the relay test vs. "exact" model (Usually not accessible. Herein: Analytically derived)
    a2, a1, a0, a0D, b0, b0D, tau, tau0, theta, w, k, T, real = symbols('a2 a1 a0 a0D b0 b0D tau tau0 theta w k T real')

    # "Exact" (physical) model parameters
    tauX = 131
    b0X = -2.146 * 10 ** (-7)
    b0DX = 2.334 * 10 ** (-6)
    tau0X = 1.5
    a2X = 0.1767
    a1X = 0.009
    a0X = 1.413 * 10 ** (-4)
    a0DX = -7.624 * 10 ** (-5)
    thetaX = 143

    K = (b0X + b0DX) / (a0X + a0DX)  # Static gain (can be measured or estimated)

    # Identified model parameters: XXX=[b0D tau0 tau a2 a1 a0 a0D theta]
    XXX = obtained_values
    tau_x = XXX[2]
    b0D_x = XXX[0]
    tau0_x = XXX[1]
    a2_x = XXX[3]
    a1_x = XXX[4]
    a0_x = XXX[5]
    a0D_x = XXX[6]
    theta_x = XXX[7]
    b0_x = K * (a0_x + a0D_x) - b0D_x

    ReG = ((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2)) \
          / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2) \
          + ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3)) \
          / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)

    ImG = ((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3)) \
          / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2) \
          - ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2)) \
          / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)

    omega = np.linspace(0, 0.1, 100)

    ReG_ = ReG.subs([(b0, b0X), (b0D, b0DX), (tau0, tau0X), (tau, tauX), (a2, a2X), (a1, a1X), (a0, a0X), (a0D, a0DX), (theta, thetaX)])
    ImG_ = ImG.subs([(b0, b0X), (b0D, b0DX), (tau0, tau0X), (tau, tauX), (a2, a2X), (a1, a1X), (a0, a0X), (a0D, a0DX), (theta, thetaX)])
    ReG_lambda = lambdify(w, ReG_)
    ImG_lambda = lambdify(w, ImG_)
    ReGX = []
    ImGX = []
    for w_value in omega:
        ReGX.append(ReG_lambda(w_value))
        ImGX.append(ImG_lambda(w_value))

    ReG__ = ReG.subs([(b0, b0_x), (b0D, b0D_x), (tau0, tau0_x), (tau, tau_x), (a2, a2_x), (a1, a1_x), (a0, a0_x), (a0D, a0D_x), (theta, theta_x)])
    ImG__ = ImG.subs([(b0, b0_x), (b0D, b0D_x), (tau0, tau0_x), (tau, tau_x), (a2, a2_x), (a1, a1_x), (a0, a0_x), (a0D, a0D_x), (theta, theta_x)])
    ReG__lambda = lambdify(w, ReG__)
    ImG__lambda = lambdify(w, ImG__)
    ReG_x = []
    ImG_x = []
    for w_value in omega:
        ReG_x.append(ReG__lambda(w_value))
        ImG_x.append(ImG__lambda(w_value))

    resultX = pd.DataFrame(list(zip(ReGX, ImGX)), columns=['ReGX', 'ImGX'])
    result_x = pd.DataFrame(list(zip(ReG_x, ImG_x)), columns=['ReG_x', 'ImG_x'])
    fig, ax = plt.subplots(figsize=(16, 6))
    resultX.plot.scatter(x="ReGX", y="ImGX", s=5.0, title="Physical model results vs obtained model results", label="physical", ax=ax)
    result_x.plot.scatter(x="ReG_x", y="ImG_x", c="red", s=5.0, label="obtained", ax=ax)
    plt.legend(loc="upper left")
    plt.show()

    return resultX, result_x
