# *- coding: utf-8 -*-
"""
Created on Fri May 20 22:52:05 2022

@author: William GENETELLI
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
tkick = 10 #20
K = 30

p = np.linspace(start=-N, stop=N, endpoint=False, num=2*N)
x = (2*np.pi)/(2*N) * p  

f    = np.zeros(2*N)   # setting up initial distribution
f[np.where(p==0)] = 1  # Dirac initial distribution in p = 0

psi_init_p = np.zeros((2*N,2))
psi_init_p[:,0] = f
# norm = np.linalg.norm(psi_init_p)
temp = np.zeros(4*N)
temp[:2*N] = psi_init_p[:,0]
temp[2*N:] = psi_init_p[:,1]
psi_init_p = temp

# omega = 11
# alpha = 0
# epsilon = 0.8
# kb = 2.89

# sim = kr.SpinKickedRotorRA(omega,alpha,epsilon,kb)
    
# psi_final = sim.loop(x, p, psi_init_p, K, 0, 10)

# plt.figure()
# plt.plot(p,psi_final, label=f'Ω={omega}, α={alpha}, ϵ={epsilon}, kb={kb}, K={K}')

epsilon = 0.3
mu = 0.8
kb = 1

sim2        = kr.SpinKickedRotor(mu, epsilon, kb)
Uk = sim2.Ukick(x,K)
ping = time.time()
psi_final   = sim2.avgPsi(x, p, psi_init_p, K, tkick, 200)
pong = time.time()
print('Psi', pong-ping)
plt.figure()
plt.plot(p,psi_final)

# E = np.linspace(start=0.1, stop=1, num=10)
# M = np.linspace(start=0.1, stop=1, num=10)

# for e in E:
#     plt.figure()
#     for m in M:
#         mu          = m
#         epsilon     = e
#         sim2        = kr.SpinKickedRotor(kb, mu, epsilon)
#         psi_final   = sim2.avgPsi(x, p, psi_init_p, K, tkick, 1000)
        
#         plt.plot(p, psi_final, label=f'ϵ={epsilon}, μ={mu}, kb={kb}, K={K}')

## the mean and std of the data = mu and sigma


