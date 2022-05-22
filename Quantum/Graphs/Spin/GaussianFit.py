# -*- coding: utf-8 -*-
"""
Created on Sun May 22 12:06:45 2022

@author: ChrisZeThird
"""
import NumericalApplications2 as na

from scipy import optimize as opt
import scipy.stats as stats
import numpy as np

psi_final = na.psi_final
p = na.p

# X,Y are the points of your PDF

def gaussian(sigma, mu, x):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

def loss_fun(params):
    mu, sigma = params
    return ((stats.norm.pdf(p, mu, sigma) - psi_final)**2).mean()

initial_guess = np.array([0, 70])  # initial guess, just needs to be close-enough to the true values
res = opt.minimize(loss_fun, x0=initial_guess)

mu, sigma = res.x

g = gaussian(sigma, mu, p)
g0 = g[np.where(p==0)]
g[np.where(p==0)] = g0/2

plt.figure()
plt.plot(p, psi_final)
plt.plot(p, g, 'r', label='fit')

ax = plt.gca()
ax.set_xlim([-300, 300])

plt.title('Density of probability as a function of p \n in the symplectic case', fontsize=15)
plt.ylabel('|Ψ|²')
plt.xlabel('p')

plt.show()