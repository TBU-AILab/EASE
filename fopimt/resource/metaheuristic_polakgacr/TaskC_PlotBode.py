from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sympy import *
from sympy.utilities.lambdify import lambdify


def plot_bode_comparison(obtained_values):
    # Bode plots comaparison
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

    AbsG = ((b0 ** 2 + 2 * cos(tau0 * w) * b0 * b0D + b0D ** 2)
            / (a0 ** 2 - 2 * a1 * w ** 4 + a0D ** 2 + w ** 6 + a1 ** 2 * w ** 2 + a2 ** 2 * w ** 4 + 2 * a0D * w ** 3 * sin(theta * w)
               + 2 * a0 * a0D * cos(theta * w) - 2 * a0 * a2 * w ** 2 - 2 * a1 * a0D * w * sin(theta * w) - 2 * a2 * a0D * w ** 2 * cos(theta * w))
            ) ** (1 / 2)
    # Chci Ra*AbsG=1
    PhaseG = atan2(((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)
                   - ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)
                   , ((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)
                   + ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2))
    # We want: PhaseG=-pi+wu*tau_plus, AbsG*Ra=1

    omega = np.linspace(0, 0.1, 101)

    AbsG_ = AbsG.subs([(b0, b0X), (b0D, b0DX), (tau0, tau0X), (tau, tauX), (a2, a2X), (a1, a1X), (a0, a0X), (a0D, a0DX), (theta, thetaX)])
    PhaseG_ = PhaseG.subs([(b0, b0X), (b0D, b0DX), (tau0, tau0X), (tau, tauX), (a2, a2X), (a1, a1X), (a0, a0X), (a0D, a0DX), (theta, thetaX)])
    AbsG_lambda = lambdify(w, AbsG_)
    PhaseG_lambda = lambdify(w, PhaseG_)
    AbsGX = []
    PhaseGX = []

    phase_previous = 0.0
    pi_counter = 0

    for w_value in omega:
        phase_current = PhaseG_lambda(w_value)
        if phase_current - (pi_counter + 1) * pi > phase_previous:
            pi_counter += 2
        phase_current -= pi_counter * pi
        phase_previous = phase_current

        AbsGX.append(AbsG_lambda(w_value))
        PhaseGX.append(phase_current)

    AbsG__ = AbsG.subs([(b0, b0_x), (b0D, b0D_x), (tau0, tau0_x), (tau, tau_x), (a2, a2_x), (a1, a1_x), (a0, a0_x), (a0D, a0D_x), (theta, theta_x)])
    PhaseG__ = PhaseG.subs([(b0, b0_x), (b0D, b0D_x), (tau0, tau0_x), (tau, tau_x), (a2, a2_x), (a1, a1_x), (a0, a0_x), (a0D, a0D_x), (theta, theta_x)])
    AbsG__lambda = lambdify(w, AbsG__)
    PhaseG__lambda = lambdify(w, PhaseG__)
    AbsG_x = []
    PhaseG_x = []

    phase_previous = 0.0
    pi_counter = 0

    for w_value in omega:
        phase_current = PhaseG__lambda(w_value)
        if phase_current - (pi_counter + 1) * pi > phase_previous:
            pi_counter += 2
        phase_current -= pi_counter * pi
        phase_previous = phase_current

        AbsG_x.append(AbsG__lambda(w_value))
        PhaseG_x.append(phase_current)

    resultX = pd.DataFrame(list(zip(AbsGX, PhaseGX)), columns=['AbsGx', 'PhaseGX'])
    result_x = pd.DataFrame(list(zip(AbsG_x, PhaseG_x)), columns=['AbsG_x', 'PhaseG_x'])

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

    ax1.scatter(x=omega, y=AbsGX, s=5.0, label="physical")
    ax1.scatter(x=omega, y=AbsG_x, s=5.0, c="red", label="obtained")
    ax1.set_title("Physical model vs obtained model absolute")
    ax1.set_xlabel('frequency')
    ax1.set_ylabel('absolute value')
    ax1.legend(loc="upper right")

    ax2.scatter(x=omega, y=PhaseGX, s=5.0, label="physical")
    ax2.scatter(x=omega, y=PhaseG_x, s=5.0, c="red", label="obtained")
    ax2.set_title("Physical model vs obtained model phase")
    ax2.set_xlabel('frequency')
    ax2.set_ylabel('phase value')
    ax2.legend(loc="upper left")

    plt.show()

    return resultX, result_x


def plot_bode_fitting(obtained_values):
    a2, a1, a0, a0D, b0, b0D, tau, tau0, theta, w, k, T = symbols('a2 a1 a0 a0D b0 b0D tau tau0 theta w k T', real=True)

    K = 0.0322  # Static gain estimation

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

    AbsG = ((b0 ** 2 + 2 * cos(tau0 * w) * b0 * b0D + b0D ** 2)
            / (a0 ** 2 - 2 * a1 * w ** 4 + a0D ** 2 + w ** 6 + a1 ** 2 * w ** 2 + a2 ** 2 * w ** 4 + 2 * a0D * w ** 3 * sin(theta * w)
               + 2 * a0 * a0D * cos(theta * w) - 2 * a0 * a2 * w ** 2 - 2 * a1 * a0D * w * sin(theta * w) - 2 * a2 * a0D * w ** 2 * cos(theta * w))
            ) ** (1 / 2)
    # Chci Ra*AbsG=1
    PhaseG = atan2(((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)
                   - ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)
                   , ((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)
                   + ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2))
    # We want: PhaseG=-pi+wu*tau_plus, AbsG*Ra=1

    omega = [0.0002, 0.0003, 0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002, 0.003, 0.005, 0.008, 0.01, 0.011, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025]

    AbsG__ = AbsG.subs([(b0, b0_x), (b0D, b0D_x), (tau0, tau0_x), (tau, tau_x), (a2, a2_x), (a1, a1_x), (a0, a0_x), (a0D, a0D_x), (theta, theta_x)])
    PhaseG__ = PhaseG.subs([(b0, b0_x), (b0D, b0D_x), (tau0, tau0_x), (tau, tau_x), (a2, a2_x), (a1, a1_x), (a0, a0_x), (a0D, a0D_x), (theta, theta_x)])

    AbsG__lambda = lambdify(w, AbsG__)
    PhaseG__lambda = lambdify(w, PhaseG__)
    AbsG_x = []
    PhaseG_x = []

    phase_previous = 0.0
    pi_counter = 0

    for w_value in omega:
        phase_current = PhaseG__lambda(w_value)
        if phase_current - (pi_counter + 1) * pi > phase_previous:
            pi_counter += 2
        phase_current -= pi_counter * pi
        phase_previous = phase_current

        AbsG_x.append(AbsG__lambda(w_value))
        PhaseG_x.append(phase_current)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

    ax1.scatter(x=omega, y=AbsG_x, s=10.0, c="green", label="obtained")
    ax1.set_title("Data vs obtained model absolute")
    ax1.set_xlabel('frequency')
    ax1.set_ylabel('absolute value')
    ax2.scatter(x=omega, y=PhaseG_x, s=10.0, c="green", label="obtained")
    ax2.set_title("Data vs obtained model phase")
    ax2.set_xlabel('frequency')
    ax2.set_ylabel('phase value')

    ax1.scatter([0.0002, 0.0003, 0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002, 0.003, 0.005, 0.008, 0.01, 0.011, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025],
                [0.0325043074068653, 0.0324085559690647, 0.0321284998093593, 0.0314696885907694, 0.0308970111823134, 0.0302361538559388, 0.0291281736468320, 0.0279356725353087, 0.0271188071271581, 0.0231276566041612, 0.0171394807389256,
                 0.0123438243668646, 0.0106190442131107, 0.0100300199401596, 0.00956026150269960, 0.00893456770079001, 0.00863177849576783, 0.00858676306881703, 0.00879860216170728, 0.0104117241607718],
                c="red", label="data", s=5.0)

    ax2.scatter([0.0002, 0.0003, 0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002, 0.003, 0.005, 0.008, 0.01, 0.011, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025],
                [-0.0874845873048167, -0.131205775304715, -0.217723670844056, -0.344562959790005, -0.426439991454951, -0.505813570952377, -0.619810486928930, -0.727147706434357, -0.795045843197777, -1.09322581166291, -1.53753364335463,
                 -2.02357479861843, -2.29556176344770, -2.42604532835201, -2.55271972824282, -2.80391424273776, -3.05575756637431, 2.96955149485742 - 2*pi, 2.69878595145307 - 2*pi, 1.90955617624874 - 2*pi],
                c="red", label="data", s=5.0)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")
    plt.show()


def plot_bode_frequencies(new_frequencies):
    # Bode plots comaparison
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

    AbsG = ((b0 ** 2 + 2 * cos(tau0 * w) * b0 * b0D + b0D ** 2)
            / (a0 ** 2 - 2 * a1 * w ** 4 + a0D ** 2 + w ** 6 + a1 ** 2 * w ** 2 + a2 ** 2 * w ** 4 + 2 * a0D * w ** 3 * sin(theta * w)
               + 2 * a0 * a0D * cos(theta * w) - 2 * a0 * a2 * w ** 2 - 2 * a1 * a0D * w * sin(theta * w) - 2 * a2 * a0D * w ** 2 * cos(theta * w))
            ) ** (1 / 2)
    # Chci Ra*AbsG=1
    PhaseG = atan2(((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)
                   - ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)
                   , ((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)
                   + ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3))
                   / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2))
    # We want: PhaseG=-pi+wu*tau_plus, AbsG*Ra=1

    omega = np.linspace(0, 0.1, 101)

    AbsG_ = AbsG.subs([(b0, b0X), (b0D, b0DX), (tau0, tau0X), (tau, tauX), (a2, a2X), (a1, a1X), (a0, a0X), (a0D, a0DX), (theta, thetaX)])
    PhaseG_ = PhaseG.subs([(b0, b0X), (b0D, b0DX), (tau0, tau0X), (tau, tauX), (a2, a2X), (a1, a1X), (a0, a0X), (a0D, a0DX), (theta, thetaX)])
    AbsG_lambda = lambdify(w, AbsG_)
    PhaseG_lambda = lambdify(w, PhaseG_)
    AbsGX = []
    PhaseGX = []

    phase_previous = 0.0
    pi_counter = 0

    for w_value in omega:
        phase_current = PhaseG_lambda(w_value)
        if phase_current - (pi_counter + 1) * pi > phase_previous:
            pi_counter += 2
        phase_current -= pi_counter * pi
        phase_previous = phase_current

        AbsGX.append(AbsG_lambda(w_value))
        PhaseGX.append(phase_current)

    AbsG_x = []
    PhaseG_x = []

    phase_previous = 0.0
    pi_counter = 0

    for w_value in new_frequencies:
        phase_current = PhaseG_lambda(w_value)
        if phase_current - (pi_counter + 1) * pi > phase_previous:
            pi_counter += 2
        phase_current -= pi_counter * pi
        phase_previous = phase_current

        AbsG_x.append(AbsG_lambda(w_value))
        PhaseG_x.append(phase_current)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

    ax1.scatter(x=new_frequencies, y=AbsG_x, s=30.0, c="red", label="new frequencies")
    ax1.scatter(x=omega, y=AbsGX, s=5.0, label="physical model")
    ax1.set_title("New frequencies marked on physical model")
    ax1.set_xlabel('frequency')
    ax1.set_ylabel('absolute value')
    ax1.legend(loc="upper right")

    plt.show()