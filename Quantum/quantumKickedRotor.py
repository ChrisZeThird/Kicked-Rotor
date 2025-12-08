# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:19:44 2022

@author: ChrisZeThird
"""
from numpy import pi, cos, exp, diag, dot, zeros, random, ndarray
import numpy.fft as fft


class KickedRotor:
    
    def __init__(self, kb=2.89, optRKR=False):
        self.I = complex(0, 1)
        self.kb = kb
        self.optRKR = optRKR
        
    def Ukick(self, x: ndarray, K: float, f: ndarray) -> ndarray:
        """Input : x -> array, impulsion
                   K -> float, kick's strength
                   f -> array, state before the Kick
           Output : array, returns the state after a Kick"""
        
        k = K/self.kb

        Uk = diag(exp(-self.I*k*cos(x)))
        f_fft = fft.fftshift(fft.fft(fft.ifftshift(f)))
        
        return dot(Uk, f_fft)
        
    def Uprop(self, p: ndarray, b: float, fk: ndarray) -> ndarray:
        """Input : p -> array, impulsion
                   b -> float, pseudo-impulsion
                   fk -> array, state after the Kick
           Output : array, returns the state after a Propagation"""
           
        L = len(p)
        
        if self.optRKR:
            phiVect = 2*pi*random.rand(L)
            Up = diag(exp(-self.I*phiVect))
        else:
            Up = diag(exp(-(self.I*self.kb/2) * ((p + b)**2)))
            
        fp = fft.ifftshift(fft.ifft(fft.fftshift(fk)))
        return dot(Up, fp)
    
    def loop(self, x: ndarray, p: ndarray, Psi: ndarray, K: float, nkick: int, b: float):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 1d array, initial states
                   K -> float, kicks strength
                   nkick -> int, number of kicks
                   b -> float, pseudo-impulsion
           Output : array, returns a state after one single iteration """

        res = Psi
        for j in range(nkick):
            Uk = self.Ukick(x, K, res)
            res = self.Uprop(p, b, Uk)
    
        return res
    
    def avgPsi(self, x: ndarray, p: ndarray, Psi: ndarray, K: float, nkick: int, navg: int):
        """Input : x -> array, positions 
                   p -> array, impulsion
                   Psi -> 1d array, initial states
                   K -> float, kicks strength
                   nkick -> int, number of iterations of the simulation
                   navg -> int, number of beta (and omega) values to average on
           Output : array, returns the average density of probability for n_beta values of the pseudo-impulsion"""
        
        L = len(x)
        average = zeros((L, navg))
        
        for i in range(navg):
            b = random.uniform(low=-0.5, high=0.5)
            psi_final = self.loop(x, p, Psi, K, nkick, b)
            average[:, i] = abs(psi_final)**2
            if i % 50 == 0:
                print(f'loop {i}')
        
        return average(average, axis=1)
