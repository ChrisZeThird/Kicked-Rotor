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
N = 2**m
#b = 0
tkick =  10 #20
K = 50

p = np.linspace(start=-N, stop=N, endpoint=False, num=2*N)
x = (2*np.pi)/(N) * p  

f    = np.zeros(2*N)   # setting up initial distribution
f[np.where(p==0)] = 1  # Dirac initial distribution in p = 0

psi_init_p = np.zeros((2*N,2))
psi_init_p[:,0] = f
        
temp = np.zeros(4*N)
temp[:2*N] = psi_init_p[:,0]
temp[2*N:] = psi_init_p[:,1]
psi_init_p = temp

epsilon = 0.3
mu = 0.8
kb = 2.89
nbeta = 100

sim = kr.SpinKickedRotor(mu, epsilon, kb)
psi_final = sim2.avgPsi(x, p, psi_init_p, K, tkick, nbeta)

## Plot only for the first Hamiltonien

plt.plot(p,psi_final, label=f'μ={mu}, ϵ={epsilon}, kb={kb}, K={K}, nkick={tkick}, nbeta={nbeta}')

