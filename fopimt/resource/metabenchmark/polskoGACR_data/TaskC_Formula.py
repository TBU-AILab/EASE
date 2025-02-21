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

M = 20  # Number of estimated frequency points
f = (ReG - A) ** 2 + (ImG - B) ** 2  # Cost fun (unconstrained) - general

# Some points determined (approximately)
# Particular frequencies
wx = [0.0002, 0.0003, 0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002, 0.003, 0.005, 0.008, 0.01, 0.011, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025]
# Real parts
Ax = [0.03238, 0.03213, 0.03137, 0.02962, 0.02813, 0.02645, 0.023710, 0.02087, 0.01899, 0.01063, 0.00057, -0.0054, -0.00704, -0.00757, -0.00795, -0.00843, -0.00860, -0.00846, -0.00795, -0.00346]
# Imaginary parts
Bx = [-0.00284, -0.00424, -0.00694, -0.01063, -0.01278, -0.01465, -0.01692, -0.01857, -0.01936, -0.02054, -0.01713, -0.01110, -0.00795, -0.00658, -0.00531, -0.00296, -0.00074, 0.00147, 0.00377, 0.00982]

f_lambda = lambdify([b0D, tau0, tau, a2, a1, a0, a0D, theta, w, A, B], f)


def evaluate(w):
    errors = []
    for i in range(0, M):
        errors.append(f_lambda(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], wx[i], Ax[i], Bx[i]))
    fitness = sum(errors)
    return fitness
