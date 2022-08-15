# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:19:44 2022

@author: ChrisZeThird
"""
import numpy as np
from scipy.linalg import expm
import numpy.fft as fft
import matplotlib.pyplot as plt

import time

class KickedRotor():
    
    def __init__(self, kb = 2.89, optRKR=False):   
        self.I  = complex(0,1)
        self.kb = kb
        self.optRKR = optRKR
        
    def Ukick(self, x, K, f):
        """Input : x -> array, impulsions
                   K -> float, kick's strength
           Output : array, returns the state after a Kick"""
        
        k = K/self.kb

        Uk = np.diag(np.exp(-self.I*k*np.cos(x)))
        f_fft = fft.fftshift(fft.fft(fft.ifftshift(f)))
        
        return np.dot(Uk, f_fft)
        
    def Uprop(self, p, b, fk):
        """Input : p -> array, impulsions
                   b -> float, pseudo-impulsion
           Output : array, returns the state after a Propagation"""
           
        L = len(p)
        
        if self.optRKR:
            phiVect = 2*np.pi*np.random.rand(L)
            Up =  np.diag(np.exp(-self.I*phiVect))
        else:
            Up =  np.diag(np.exp(-(self.I*self.kb/2) * ((p + b)**2)))  
            
        fp = fft.ifftshift(fft.ifft(fft.fftshift(fk)))
        return np.dot(Up, fp)
    
    def loop(self, x, p, Psi, K, nkick, b):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 1d array, initial states
                   K -> float, kicks strength
                   nkick -> int, number of kicks
                   b -> float, pseudo-impulsion
           Output : array, returns a state after one single iteration """
        
        L = len(x)
        res = Psi
        for j in range(nkick):
            Uk = self.Ukick(x,K,res)
            res = self.Uprop(p,b,Uk)            
    
        return res
    
    def avgPsi(self, x, p, Psi, K, nkick, navg):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 1d array, initial states
                   K -> float, kicks strength
                   nkick -> int, number of iterations of the simulation
                   navg -> int, number of beta (and omega) values to average on
           Output : array, returns the average density of probability for n_beta values of the pseudo-impulsion"""
        
        L = len(x)
        average = np.zeros((L,navg))   
        
        for i in range(navg):
            b = np.random.uniform(low=-0.5, high=0.5)
            psi_final = self.loop(x, p, Psi, K, nkick,b)
            average[:,i] = abs(psi_final)**2
            if i % 50 == 0:
                print(f'loop {i}')
        
        return np.average(average, axis=1)
