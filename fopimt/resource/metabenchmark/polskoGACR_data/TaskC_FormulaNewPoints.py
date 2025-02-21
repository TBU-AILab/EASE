import ast

from sympy import *

s = symbols('s')
a2, a1, a0, a0D, b0, b0D, tau, tau0, theta, w, k, A, B = symbols('a2 a1 a0 a0D b0 b0D tau tau0 theta w k A B', real=True)

# Open-loop model transfer function representations in the frequency domain
ReG = ((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2)) \
      / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2) \
      + ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3)) \
      / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)

ImG = ((b0D * cos(w * (tau + tau0)) + b0 * cos(tau * w)) * (a0D * sin(theta * w) - a1 * w + w ** 3)) \
      / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2) \
      - ((b0D * sin(w * (tau + tau0)) + b0 * sin(tau * w)) * (a0 + a0D * cos(theta * w) - a2 * w ** 2)) \
      / ((a0 + a0D * cos(theta * w) - a2 * w ** 2) ** 2 + (a0D * sin(theta * w) - a1 * w + w ** 3) ** 2)

# We want: Ra*ReG=-1, Ra*ImG=0. "tau_plus" is simply modulated as the increase of "tau"
# The controlled model itself does not contain "tau_plus"!!!

# Substitution k=...
ReG = ReG.subs(b0, k * (a0 + a0D) - b0D)
ImG = ImG.subs(b0, k * (a0 + a0D) - b0D)

# --- MULTIPLE-POINT DATA ---
K = 0.03257608361512449  # Static gain estimation
ReG = ReG.subs(k, K)
ImG = ImG.subs(k, K)  # Particular static gain value

f = (ReG - A) ** 2 + (ImG - B) ** 2  # Cost fun (unconstrained) - general


with open('fopimt/resource/metaheuristic_polakgacr/utils/new_points.txt', 'r') as dataFile:
    lines = dataFile.readlines()
    # Some points determined (from bode plots physical model)
    # Particular frequencies
    wx = ast.literal_eval(lines[0])
    # Real parts
    Ax = ast.literal_eval(lines[1])
    # Imaginary parts
    Bx = ast.literal_eval(lines[2])

f_lambda = lambdify([b0D, tau0, tau, a2, a1, a0, a0D, theta, w, A, B], f)


def evaluate(w):
    errors = []
    for i in range(0, len(wx)):
        errors.append(f_lambda(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], wx[i], Ax[i], Bx[i]))
    fitness = sum(errors)
    return fitness
