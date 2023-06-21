# -*- coding: utf-8 -*-
"""
Created on Sun May 22 12:06:45 2022

@author: ChrisZeThird
"""
import KickedRotor2 as kr

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

## Import parameters

N = 1024
tkick = 10

p = np.linspace(start=-N, stop=N, endpoint=False, num=2*N)
x = (2*np.pi)/(2*N) * p  

f = np.zeros(2*N) # setting up initial distribution
f[N] = 1             # Dirac initial distribution in p = 0

psi_init_p = np.zeros((2, 2*N))
psi_init_p[0, :] = f
norm = np.linalg.norm(psi_init_p)
psi_init_p = psi_init_p/norm

omega = 10
alpha = 0
epsilon = 0.9

# epsilon = 0.3 #*np.sqrt(5)
# mu = 0.6 #*np.sqrt(5)
kb = 2.89
K = 55
n_avg = 1000

# Computes model

# sim        = kr.SpinKickedRotor(kb, mu, epsilon)
# psi_final1 = sim.avgPsi(x, p, psi_init_p, K, tkick-1, n_beta)
# psi_final2 = sim.avgPsi(x, p, psi_init_p, K, tkick, n_beta) # checks stability of the solution
# psi_final3 = sim.avgPsi(x, p, psi_init_p, K, tkick+1, n_beta)

ydata = np.zeros(len(p))
sim = kr.SpinKickedRotorRA(omega, alpha, epsilon, kb)
for l in range(100):    
    ydata += sim.avgPsi(x, p, psi_init_p, K, tkick, n_avg)
 
ydata /= 100    
xdata = p

# ydata1 = psi_final1
# ydata2 = psi_final2
# ydata3 = psi_final3


# Fitting function

def fit_gaussien(q, A, c):
    return A * np.exp(-q**2 / c)


popt, pcov = curve_fit(fit_gaussien, xdata, ydata)
function = fit_gaussien(xdata, *popt)
function[np.where(p == 0)] = function[1024] / 2.
err_rel = np.abs(ydata[1024] - function[1024]) / ydata[1024]

# popt1,pcov1 = curve_fit(fit_gaussien,xdata,ydata1)
# function1   = fit_gaussien(xdata,*popt1)

# popt2,pcov2 = curve_fit(fit_gaussien,xdata,ydata2)
# function2   = fit_gaussien(xdata,*popt2)

# popt3,pcov3 = curve_fit(fit_gaussien,xdata,ydata3)
# function3   = fit_gaussien(xdata,*popt3)

# function1[np.where(p==0)]=function1[1024]/2.
# function2[np.where(p==0)]=function2[1024]/2.
# function3[np.where(p==0)]=function3[1024]/2.

# err_rel1 = np.abs(ydata1[1024]-function1[1024])/ydata1[1024]
# err_rel2 = np.abs(ydata2[1024]-function2[1024])/ydata2[1024]
# err_rel3 = np.abs(ydata3[1024]-function3[1024])/ydata3[1024]

# Plot

fig = plt.figure()
fig.suptitle(f'Density of probability as a function of p in the symplectic case', fontsize=20)

ax1 = fig.add_subplot(111)
ax1.set_title(f'tkick={tkick}')
ax1.set_ylabel('|Ψ|²')
ax1.set_xlabel('p')
ax1.plot(xdata, ydata, label=f'Ω={omega}, α={alpha}, ϵ={epsilon}, kb={kb}, K={K}')
ax1.plot(xdata, function, alpha=0.5, label='fit')

ax = plt.gca()
ax.set_xlim([-300, 300])

ax1.legend()

plt.show()

# fig = plt.figure()
# fig.suptitle(f'Density of probability as a function of p in the symplectic case \n ϵ={epsilon}, μ={mu}, kb={kb}, K={K}', fontsize=20)

# ax1 = fig.add_subplot(131)
# ax1.set_title(f'tkick={tkick-1}')
# ax1.set_ylabel('|Ψ|²')
# ax1.set_xlabel('p')
# ax1.plot(xdata, ydata1, label=f'Final State')
# ax1.plot(xdata,function1, label='fit')
# ax1.legend()

# ax2 = fig.add_subplot(132)
# ax2.set_title(f'tkick={tkick}')
# ax2.set_ylabel('|Ψ|²')
# ax2.set_xlabel('p')
# ax2.plot(xdata, ydata2, label=f'Final State')
# ax2.plot(xdata,function2, label='fit')
# ax2.legend()

# ax3 = fig.add_subplot(133)
# ax3.set_title(f'tkick={tkick+1}')
# ax3.set_ylabel('|Ψ|²')
# ax3.set_xlabel('p')
# ax3.plot(xdata, ydata3, label='Final State')
# ax3.plot(xdata,function3, label='fit')
# ax3.legend()

# # plt.tight_layout()
# plt.show()
