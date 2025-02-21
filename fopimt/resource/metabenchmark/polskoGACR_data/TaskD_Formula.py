from sympy import *
from math import pi
import numpy as np

s = symbols('s')
a2, a1, a0, a0D, b0, b0D, tau, tau0, theta, w, a, A, B = symbols('a2 a1 a0 a0D b0 b0D tau tau0 theta w a A B', real=True)

# ---- PROCESS MODEL(S) ----
# DECAYED Process model representation
# Ga=(b0+b0D*exp(-tau0*(s+a)))*exp(-tau*(s+a))/((s+a)^3+a2*(s+a)^2+a1*(s+a)+a0+a0D*exp(-theta*(s+a)))
ReGa = (b0 * exp(-a * tau) * cos(tau * w) * (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(-a * theta) * cos(theta * w))) \
       / ((a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w) ** 2
          + (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(-a * theta) * cos(theta * w)) ** 2) \
       - (b0 * exp(-a * tau) * sin(tau * w) * (a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w)) \
       / ((a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w) ** 2
          + (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(-a * theta) * cos(theta * w)) ** 2) \
       - (b0D * exp(-a * (tau + tau0)) * sin(w * (tau + tau0)) * (a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w)) \
       / ((a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w) ** 2
          + (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(-a * theta) * cos(theta * w)) ** 2) + \
       (b0D * exp(-a * (tau + tau0)) * cos(w * (tau + tau0)) * (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(-a * theta) * cos(theta * w))) \
       / ((a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w) ** 2
          + (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(-a * theta) * cos(theta * w)) ** 2)

ImGa = -(b0D * exp(-a * (tau + tau0)) * sin(w * (tau + tau0)) * (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(a * theta) * cos(theta * w))) \
       / ((a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w) ** 2
          + (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(-a * theta) * cos(theta * w)) ** 2) \
       - (b0 * exp(-a * tau) * cos(tau * w) * (a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w)) \
       / ((a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w) ** 2
          + (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(-a * theta) * cos(theta * w)) ** 2) \
       - (b0 * exp(-a * tau) * sin(tau * w) * (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(-a * theta) * cos(theta * w))) \
       / ((a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w) ** 2
          + (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(-a * theta) * cos(theta * w)) ** 2) \
       - (b0D * exp(-a * (tau + tau0)) * cos(w * (tau + tau0)) * (a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w)) \
       / ((a1 * w + 3 * a ** 2 * w - w ** 3 - a0D * exp(-a * theta) * sin(theta * w) + 2 * a * a2 * w) ** 2
          + (a0 + a * a1 + a ** 2 * a2 - 3 * a * w ** 2 - a2 * w ** 2 + a ** 3 + a0D * exp(-a * theta) * cos(theta * w)) ** 2)

# Process model- Evaluation
ax = 0.02
ReGa = ReGa.subs(a, ax)
ImGa = ImGa.subs(a, ax)

M = 20  # Number of estimated frequency points
f = (ReGa - A) ** 2 + (ImGa - B) ** 2  # Cost fun (unconstrained) - general

# "Exact" (physical) model parameters
# tauX=131; b0X=-2.146*10**(-7); b0DX=2.334*10**(-6); tau0X=1.5;
# a2X=0.1767; a1X=0.009; a0X=1.413*10**(-4); a0DX=-7.624*10**(-5); thetaX=143;

# Particular frequencies
N = 10001
dt = 0.1
# I.e., t0=0s, t1=1000s
omega = np.linspace(1, 20, 20)  # 20 samples
wx = []

for w_value in omega:
    wx.append(2 * pi * w_value / (N * dt))  # Angular frequencies

# ---- MEASURED DATA ----
# All the data correspond to frequencies "wx" !!!
# Data for model (Re, Im)
Ax = [0.000162949349611661, -0.000209549522262491, -0.000325776536422918, -0.000106390414163729, 0.000181689959173021, 0.000245423759720420, 7.65521728118226e-05, -0.000117107093159764, -0.000170004512705908, -7.51791720759046e-05,
      5.72310875425798e-05, 0.000114728817186622, 7.22982489139775e-05, -1.32663495115992e-05, -6.98212469586222e-05, -6.42195078486992e-05, -1.43416001852812e-05, 3.51507950632913e-05, 4.99143595830867e-05, 2.73539432660804e-05]
# Real parts
Bx = [-0.000331533625602285, -0.000281877398630466, 4.79887427610954e-05, 0.000286651979083965, 0.000212088148264001, -4.66682274115113e-05, -0.000206520990791309, -0.000154015374541277, 1.16051625746991e-05, 0.000130194690068203,
      0.000119411586529223, 1.89727134104489e-05, -7.20235305118172e-05, -8.88166867204942e-05, -3.77755525112997e-05, 2.90262573027370e-05, 6.10680069891318e-05, 4.34845296714443e-05, -6.74944162936694e-07, -3.53320906251569e-05]
# Imaginary parts

f_lambda = lambdify([b0, b0D, tau0, tau, a2, a1, a0, a0D, theta, w, A, B], f)


def evaluate(w):
    errors = []
    for i in range(0, M):
        errors.append(f_lambda(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], wx[i], Ax[i], Bx[i]))
    fitness = sum(errors)
    return fitness
