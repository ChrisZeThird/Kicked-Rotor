# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:36:50 2022

@author: ChrisZeThird
"""
import numpy as np
import matplotlib.pyplot as plt
import KickedRotor1 as kr

## Parameters

nbeta = 500 
m      = 7
N      = 2**m
K      = 10
t      = 10
kb     = 1

p = np.linspace(start=-N, stop=N, endpoint=False, num=2*N) # impulsions
x = np.pi/N * p                                    # positions

f    = np.zeros(2*N) # setting up initial distribution
f[N] = 1             # Dirac initial distribution in p = 0

## Numerical application

sim = kr.KickedRotor(kb,True)

# psi_final1    = sim.avgPsi(x, p, f, K, t, nbeta)
# plt.figure(1)
# plt.plot(p, psi_final1, label=f'K={K}, tkick={t}, kb={kb}, nbeta={nbeta}')

## Mean level spacing

def meanlvl(E):
    L = len(E)
    return (1/(L-1))*np.sum(np.diff(E))

## Operator of evolution

quasiENorm = []

m2      = 4
N2      = 2**m
K2      = 100

def lvlspacing(m,K,nbeta):
    N = 2**m
    quasiENorm = np.zeros((2*N -1,nbeta))
    Psi = np.diag(np.ones(2*N))
    p = np.linspace(start=-N, stop=N, endpoint=False, num=2*N) # impulsions
    x = np.pi/N * p                                    # positions

    for i in range(nbeta):
        b = np.random.uniform(low=-0.5, high=0.5)
        operator = np.zeros((2*N,2*N), dtype=complex) # will contain all the final state for every possible initial distribution
        
        for v in range(2*N):
            
            psi_final = sim.loop(x, p, Psi[:,v], K, 1, b)
            operator[:,v] = psi_final
                    
        val = np.linalg.eigvals(operator)
        phi_n = np.sort(np.angle(val) % (2*np.pi))
        D = np.diff(phi_n)
        avg = meanlvl(phi_n) 
    
        quasiE = D/avg    
        
        quasiENorm[:,i] = quasiE
        if i%100 == 0:  
            print(f'loop {i}') 
    
    return np.ravel(quasiENorm)
    

bigbase = lvlspacing(7,10,200)
smallbase = lvlspacing(2,10000,200)

fig = plt.figure(2)
ax1 = fig.add_subplot(121)
ax1.set_title(f'Cas Orthogonal \n K=10000, N=4', fontsize=20)

ax2 = fig.add_subplot(122)
ax2.set_title('Poisson \n K=20, N=128', fontsize=20)

ax1.hist(smallbase, 200, density=True)
ax2.hist(bigbase, 150, density=True)
plt.show()

## Level Spacing distribution

def Poisson(s):
    return np.exp(-s)

def Orthogonal(s):
    return (s*np.pi/2)*np.exp(-s**2 * np.pi/4)


s = np.linspace(start=0, stop=2*np.pi, num=2000)

ax2.plot(s, Poisson(s), label='Poisson')
ax2.legend()
ax1.plot(s,Orthogonal(s), label='CSE')
ax1.legend()

ax1.set_xlabel('s', fontsize=15)
ax1.set_ylabel('P(s)', fontsize=15)

ax2.set_xlabel('s', fontsize=15)
ax2.set_ylabel('P(s)', fontsize=15)

plt.show()
