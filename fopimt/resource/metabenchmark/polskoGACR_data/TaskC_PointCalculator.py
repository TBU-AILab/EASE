import numpy as np
import pandas as pd

from sympy import *
from sympy.utilities.lambdify import lambdify


def calculate_from_data():
    # --------------------------------------
    # Discrete-Time Fourier Transform (DTFT)
    # --------------------------------------

    # Inputs:
    # reading data file
    ScopeData = pd.read_csv('Task1C_Data_t_u_y.csv', sep=';', header='infer')

    # t, u, y - Sampled data vectors of time, plant inputs and outputs from a relay-feedback experiment, respectively
    t = ScopeData['t'].to_numpy()
    u = ScopeData['u'].to_numpy()
    y = ScopeData['y'].to_numpy()

    dt = 0.1  # Sampling period
    t0 = 0
    t1 = 1000  # Start a stop time

    # Start a stop data sample number
    n0 = round(t0 / dt + 1)
    n1 = round(t1 / dt + 1)

    t = t[n0:n1] - t[n0]  # Time axis shifted to the origin
    u = u[n0:n1]  # Plant input
    y = y[n0:n1]  # Plant output
    N = n1 - n0 + 1  # Number of data samples

    # Decayed signals
    ax = 0.02  # Decaying coefficient
    u_a = np.multiply(u, np.exp(-ax * t))
    y_a = np.multiply(y, np.exp(-ax * t))

    frequencies = []
    Gw_a = []
    # DTFT
    for k in range(0, 20):  # 20 samples are enough for a test
        wk = 2 * np.pi * k / (N * dt)  # Angular frequencies
        frequencies.append(wk)
        Uf = sum(np.multiply(u_a, np.exp(-1j * wk * t))) * dt
        Yf = sum(np.multiply(y_a, np.exp(-1j * wk * t))) * dt
        Gw_a.append(Yf / Uf)  # G(j*wk+ax) = The decayed (shifted) model data (frequency points)

    real = [gw.real for gw in Gw_a]
    imaginary = [gw.imag for gw in Gw_a]

    with open('new_points.txt', 'w') as result_file:
        result_file.write(str(frequencies) + '\n' + str(real) + '\n' + str(imaginary))


def calculate_from_physical_model():
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

    ReG = ((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2)) \
          / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2) \
          + ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3)) \
          / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)

    ImG = ((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3)) \
          / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2) \
          - ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2)) \
          / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)

    frequencies = [0.0002, 0.0003, 0.0005, 0.0008, 0.012, 0.015, 0.018, 0.02, 0.022, 0.028, 0.03, 0.031, 0.033, 0.04, 0.065, 0.08, 0.1]

    ReG_ = ReG.subs([(b0, b0X), (b0D, b0DX), (tau0, tau0X), (tau, tauX), (a2, a2X), (a1, a1X), (a0, a0X), (a0D, a0DX), (theta, thetaX)])
    ImG_ = ImG.subs([(b0, b0X), (b0D, b0DX), (tau0, tau0X), (tau, tauX), (a2, a2X), (a1, a1X), (a0, a0X), (a0D, a0DX), (theta, thetaX)])
    ReG_lambda = lambdify(w, ReG_)
    ImG_lambda = lambdify(w, ImG_)
    real = []
    imaginary = []
    for w_value in frequencies:
        real.append(ReG_lambda(w_value))
        imaginary.append(ImG_lambda(w_value))

    with open('new_points.txt', 'w') as result_file:
        result_file.write(str(frequencies) + '\n' + str(real) + '\n' + str(imaginary))


calculate_from_physical_model()
