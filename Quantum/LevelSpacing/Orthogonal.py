# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:36:50 2022

@author: ChrisZeThird
"""
import numpy as np
import matplotlib.pyplot as plt
import KickedRotor1 as kr

## Mean level spacing

def meanlvl(E):
    L = len(E)
    return (1/(L-1))*np.sum(np.diff(E))

def lvlspacing(m,K,nbeta):
    N = 2**m
    quasiENorm = np.zeros((2*N -1,nbeta))
    Psi = np.diag(np.ones(2*N)) # Contains all possible states (first component is a dirac, the second one a ket of 0, and vice versa)
    p = np.linspace(start=-N, stop=N, endpoint=False, num=2*N) # impulsions
    x = np.pi/N * p                                    # positions

    for i in range(nbeta):
        b = np.random.uniform(low=-0.5, high=0.5)
        operator = np.zeros((2*N,2*N), dtype=complex) # will contain all the final state for every possible initial distribution
        
        for v in range(2*N):
            
            psi_final = sim.loop(x, p, Psi[:,v], K, 1, b) # Depends on the simulation you're using, here it is for the orthogonal case
            operator[:,v] = psi_final
                    
        val = np.linalg.eigvals(operator)
        phi_n = np.sort(np.angle(val) % (2*np.pi)) # /!\ WARNING ADD [::2] IN THE SYMPLECTIC CASE BECAUSE OF THE KRAMER DEGENERENSCY
        D = np.diff(phi_n)
        avg = meanlvl(phi_n) 
    
        quasiE = D/avg    
        
        quasiENorm[:,i] = quasiE
        if i%100 == 0:  
            print(f'loop {i}') 
    
    return np.ravel(quasiENorm)

## Level Spacing distribution (does not include the Unitary case)

def Poisson(s):
    return np.exp(-s)

def Orthogonal(s):
    return (s*np.pi/2)*np.exp(-s**2 * np.pi/4)

def Symplectic(s):
    return (s**4 * 2**18 / 3**6 / np.pi**3)*np.exp(-s**2 * 64 / 9 /np.pi)
