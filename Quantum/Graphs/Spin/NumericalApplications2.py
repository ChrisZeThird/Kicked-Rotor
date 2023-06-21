# *- coding: utf-8 -*-
"""
Created on Fri May 20 22:52:05 2022

@author: ChrisZeThird
"""
import numpy as np
# import numpy.fft as fft
import matplotlib.pyplot as plt
import KickedRotor2 as kr
import time

##Parameters

m = 9
N = 2 ** m
# b = 0
tkick = 10  # 20
K = 30
optRKR = True

p = np.linspace(start=-N, stop=N, endpoint=False, num=2*N)
x = np.pi / N * p

f = np.zeros(2*N)   # setting up initial distribution
f[np.where(p == 0)] = 1  # Dirac initial distribution in p = 0

psi_init_p = np.zeros((2 * N, 2))
psi_init_p[:, 0] = f
        
temp = np.zeros(4*N)
temp[:2*N] = psi_init_p[:, 0]
temp[2*N:] = psi_init_p[:, 1]

epsilon = 0.3
mu = 0.8
kb = 1
nbeta = 500

sim2 = kr.SpinKickedRotor(mu, epsilon, kb, optRKR)
psi_final = sim2.avgPsi(x, p, temp, K, tkick, nbeta)

plt.figure()
plt.plot(p,psi_final, label=f'μ={mu}, ϵ={epsilon}, kb={kb}, K={K}, nkick={tkick}, nbeta={nbeta}')

