import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

from sympy import *
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.plotting import plot, plot3d

from math import pi


def plot_nyquist_comparison(obtained_values):
    # Nyquist plots (i.e., frequency-domain) comaparison
    # Heating model from the relay test vs. "exact" model (Usually not accessible. Herein: Analytically derived)
    a2, a1, a0, a0D, b0, b0D, tau, tau0, theta, w, a, A, B = symbols('a2 a1 a0 a0D b0 b0D tau tau0 theta w a A B', real=True)

    # "Exact" (physical) model parameters
    tauX=131; b0X=-2.146*10**(-7); b0DX=2.334*10**(-6); tau0X=1.5;
    a2X=0.1767; a1X=0.009; a0X=1.413*10**(-4); a0DX=-7.624*10**(-5); thetaX=143;

    # Identified model parameters: XXX=[b0D tau0 tau a2 a1 a0 a0D theta]
    XXX=obtained_values 
    tau_x=XXX[3]; b0D_x=XXX[1]; tau0_x=XXX[2]; a2_x=XXX[4]; a1_x=XXX[5]; a0_x=XXX[6]; a0D_x=XXX[7]; theta_x=XXX[8];
    b0_x=XXX[0];
    
    # ---- PROCESS MODEL(S) ----
    # DECAYED Process model representation
    # Ga=(b0+b0D*exp(-tau0*(s+a)))*exp(-tau*(s+a))/((s+a)^3+a2*(s+a)^2+a1*(s+a)+a0+a0D*exp(-theta*(s+a)))
    ReGa=(b0*exp(-a*tau)*cos(tau*w)*(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(-a*theta)*cos(theta*w)))\
    /((a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w)**2\
      +(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(-a*theta)*cos(theta*w))**2)\
    -(b0*exp(-a*tau)*sin(tau*w)*(a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w))\
    /((a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w)**2\
      +(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(-a*theta)*cos(theta*w))**2)\
    -(b0D*exp(-a*(tau+tau0))*sin(w*(tau+tau0))*(a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w))\
    /((a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w)**2\
      +(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(-a*theta)*cos(theta*w))**2)+\
    (b0D*exp(-a*(tau+tau0))*cos(w*(tau+tau0))*(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(-a*theta)*cos(theta*w)))\
    /((a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w)**2\
      +(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(-a*theta)*cos(theta*w))**2)

    ImGa=-(b0D*exp(-a*(tau+tau0))*sin(w*(tau+tau0))*(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(a*theta)*cos(theta*w)))\
    /((a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w)**2\
      +(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(-a*theta)*cos(theta*w))**2)\
    -(b0*exp(-a*tau)*cos(tau*w)*(a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w))\
    /((a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w)**2\
      +(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(-a*theta)*cos(theta*w))**2)\
    -(b0*exp(-a*tau)*sin(tau*w)*(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(-a*theta)*cos(theta*w)))\
    /((a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w)**2\
      +(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(-a*theta)*cos(theta*w))**2)\
    -(b0D*exp(-a*(tau+tau0))*cos(w*(tau+tau0))*(a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w))\
    /((a1*w+3*a**2*w-w**3-a0D*exp(-a*theta)*sin(theta*w)+2*a*a2*w)**2\
      +(a0+a*a1+a**2*a2-3*a*w**2-a2*w**2+a**3+a0D*exp(-a*theta)*cos(theta*w))**2)
    
    # Process model- Evaluation
    ax=0.02;
    ReGa=ReGa.subs(a,ax)
    ImGa=ImGa.subs(a,ax)

    # Particular frequencies
    N=10001 
    dt=0.1 
    # I.e., t0=0s, t1=1000s
    omega=np.linspace(1,20,20) # 20 samples
    wx = []

    for w_value in omega:
        wx.append(2*pi*w_value/(N*dt)) # Angular frequencies

    ReG_=ReGa.subs([(b0, b0X), (b0D, b0DX), (tau0, tau0X),  (tau, tauX), (a2, a2X), (a1, a1X), (a0, a0X), (a0D, a0DX), (theta, thetaX)]);
    ImG_=ImGa.subs([(b0, b0X), (b0D, b0DX), (tau0, tau0X),  (tau, tauX), (a2, a2X), (a1, a1X), (a0, a0X), (a0D, a0DX), (theta, thetaX)]);
    ReG_lambda = lambdify(w, ReG_)
    ImG_lambda = lambdify(w, ImG_)
    ReGX = []
    ImGX = []
    for w_value in wx:
        ReGX.append(ReG_lambda(w_value))
        ImGX.append(ImG_lambda(w_value))


    ReG__=ReGa.subs([(b0, b0_x), (b0D, b0D_x), (tau0, tau0_x),  (tau, tau_x), (a2, a2_x), (a1, a1_x), (a0, a0_x), (a0D, a0D_x), (theta, theta_x)]);
    ImG__=ImGa.subs([(b0, b0_x), (b0D, b0D_x), (tau0, tau0_x),  (tau, tau_x), (a2, a2_x), (a1, a1_x), (a0, a0_x), (a0D, a0D_x), (theta, theta_x)]);
    ReG__lambda = lambdify(w, ReG__)
    ImG__lambda = lambdify(w, ImG__)
    ReG_x = []
    ImG_x = []
    for w_value in wx:
        ReG_x.append(ReG__lambda(w_value))
        ImG_x.append(ImG__lambda(w_value))

    resultX = pd.DataFrame(list(zip(ReGX, ImGX)), columns =['ReGX', 'ImGX'])
    result_x = pd.DataFrame(list(zip(ReG_x, ImG_x)), columns =['ReG_x', 'ImG_x'])
    fig, ax = plt.subplots(figsize=(16,6))
    resultX.plot.scatter(x="ReGX", y="ImGX", s=5.0, title="Physical model results vs obtained model results", label="physical", ax=ax)
    result_x.plot.scatter(x="ReG_x", y="ImG_x", c="red", s=5.0, label="obtained", ax=ax)
    plt.legend(loc="upper left")
    plt.show()

    return resultX, result_x