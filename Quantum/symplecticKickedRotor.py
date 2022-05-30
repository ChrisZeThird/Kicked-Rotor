# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:04:49 2022

@author: ChrisZeThird
"""
"""
Two classes are defined here based on two different Hamiltoniens. The goal was to showcase an anti-CBS curve. The reason why a second Hamiltonien was defined
is because the first one wouldn't let me see the anti-CBS. But the reason was the missunderstanding of the operation. The function np.exp of numpy doesn't compute
the exponential of a matrix but the exponential of each elements of the array. For that reason, I had to redefine the function, using the scipy module function expm
which calculates the exponential of the matrix. However, the time it required was far too long, therefore a rework of the Hamilton function of the Kick was necessary
in order to reduce the duration. 
The parameter optRKR also allows you to generate random array of impulsion to simulate the Random Kicked Rotor
"""

import numpy as np
from scipy.linalg import expm
import numpy.fft as fft
import matplotlib.pyplot as plt

import time

class SpinKickedRotor():
    
    def __init__(self, mu, epsilon, kb = 2.89, optRKR=False):   
        self.I  = complex(0,1)
        self.kb = kb
        self.mu = mu
        self.epsilon = epsilon
        self.optRKR = optRKR
        
    def Ukick(self, x, K):
        """Input : p -> array, impulsions
                   Psi -> 2d array, initial state
                   K -> float, kick's strength
           Output : array, returns the state after a Kick"""
        
        k = K/self.kb
        L = len(x)
        Uk = np.zeros((2*L,2*L))
        
        s1 = np.sqrt(self.mu**2 * ((np.cos(x)))**2 + self.epsilon**2)
        s2 = k * (np.sin(x)) * s1
        
        U1 = np.exp(-(self.I*k)*np.cos(x)) * (np.cos(s2)) - self.epsilon*self.I*np.sin(s2)/s1
        U23 = np.exp(-(self.I*k)*np.cos(x)) * - self.mu*self.I*np.cos(x)*np.sin(s2)/s1
        U4 = np.exp(-(self.I*k)*np.cos(x)) * np.cos(s2) + self.epsilon*self.I*np.sin(s2)/s1
        
        Uk[:L,:L]       = np.diag(U1)
        Uk[:L,L:2*L]    = np.diag(U23)
        Uk[L:2*L,:L]    = np.diag(U23)
        Uk[L:2*L,L:2*L] = np.diag(U4)
        
        return Uk
        
    def Uprop(self, p,b=0):
        """Input : p -> array, impulsions
                   b -> float, pseudo-impulsion
           Output : array, returns the state after a Propagation"""
           
        L = len(p)
        
        # diag =  np.diag(np.exp(-(self.I*self.kb/2) * ((p + b)**2)))
        zeros = np.zeros((L,L), dtype=complex)
        
        if self.optRKR:
            phiVect = 2*np.pi*np.random.rand(L)
            diag =  np.diag(np.exp(-self.I*phiVect))
        else:
            diag =  np.diag(np.exp(-(self.I*self.kb/2) * ((p + b)**2)))    
        
        Up = np.zeros((2*L,2*L), dtype=complex)
                
        Up[:L,:L]       = diag
        Up[:L,L:2*L]    = zeros
        Up[L:2*L,:L]    = zeros
        Up[L:2*L,L:2*L] = diag
    
        return Up    
    
   def loop(self, x, p, Psi, Uk, Up, K, nkick, b):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 1d array, initial states
                   Uk -> 2d array, kick operator
                   Up -> 2d array, propagation operator
                   K -> float, kicks strength
                   nkick -> int, number of kicks
                   b -> float, pseudo-impulsion
           Output : array, returns a state after one single iteration """
        
        L = len(x)
        res = Psi
        
        ## RECURSIVE DEFINITION TO AVOID DOUBLE LOOP ON THE BETA AVERAGE CALCULATIONS
        
        if nkick == 0:
            return res
        
        elif nkick == 1:
            fk = np.zeros(2*L, dtype=complex)
            fk[:L] = fft.fftshift(fft.fft(fft.ifftshift(res[:L])))
            fk[L:] = fft.fftshift(fft.fft(fft.ifftshift(res[L:])))
            Uk_f = np.dot(Uk, fk)
            
            fp = np.zeros(2*L, dtype=complex)
            fp[:L] = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f[:L])))
            fp[L:] = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f[L:])))
            res = np.dot(Up, fp)
            return res
        
        else:
            nkick = nkick - 1
            fk = np.zeros(2*L, dtype=complex)
            fk[:L] = fft.fftshift(fft.fft(fft.ifftshift(res[:L])))
            fk[L:] = fft.fftshift(fft.fft(fft.ifftshift(res[L:])))
            Uk_f = np.dot(Uk, fk)
            
            fp = np.zeros(2*L, dtype=complex)
            fp[:L] = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f[:L])))
            fp[L:] = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f[L:])))
            res = np.dot(Up, fp)
            # print(res)
            return self.loop(x, p, res, Uk, Up, K, nkick, b)
    
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
        Uk = self.Ukick(x,K)
        
        for i in range(navg):
            b = np.random.uniform(low=-0.5, high=0.5)
            Up = self.Uprop(p,b)
            psi_final = self.loop(x, p, Psi, Uk, Up, K, nkick, b)
            average[:,i] = abs(psi_final[:L])**2 + abs(psi_final[L:])**2
            if i % 10 == 0:
                print(f'loop {i}')
        
        return np.average(average, axis=1)
    
class SpinKickedRotorRA():
    
    def __init__(self, omega, alpha, epsilon, kb = 2.89):   
        self.I  = complex(0,1)
        self.kb = kb
        self.omega = omega
        self.alpha = alpha
        self.epsilon = epsilon
        
    def Ukick(self, x, K, n_kick):
        """Input : x -> array, positions
                   Psi -> 1d array, initial states
                   K -> float, kick's strength
                   n_kick -> int, number of kicks
           Output : array, returns the state after a Kick"""
        
        L = len(x)
        res_kick = np.zeros(2*L, dtype=object)
    
        zeros = np.zeros((L,L))
        Uk = np.zeros((2*L,2*L))
        
        U1 = np.diag(np.cos(x) + self.epsilon * np.sin(x))
        U2 = zeros
        U3 = zeros
        U4 = np.diag(np.cos(x) - self.epsilon * np.sin(x))
        
        coeff = K*(1 - ((-1)**n_kick) * self.alpha)
        
        Uk[:L,:L]       = U1
        Uk[:L,L:2*L]    = U2
        Uk[L:2*L,:L]    = U3
        Uk[L:2*L,L:2*L] = U4
        
        Uk = np.exp(-(self.I/self.kb)*coeff*Uk)
        return Uk
        
    def Uprop(self, p, b=0):
        """Input : p -> array, impulsions
                   Psi -> 2d array, initial state
                   b -> float, pseudo-impulsion
           Output : array, returns the state after a Propagation"""
        
        L = len(p)
        
        identity = np.identity(L)
        
        Up = np.zeros((2*L,2*L))
        
        U1 = np.diag((p + self.kb*b)**2)
        U2 = identity*(self.omega)
        U3 = identity*(self.omega)
        U4 = np.diag((p + self.kb*b)**2 )
    
        Up[:L,:L]       = U1
        Up[:L,L:2*L]    = U2
        Up[L:2*L,:L]    = U3
        Up[L:2*L,L:2*L] = U4
        
        Up = expm(-(self.I/2/self.kb) * Up)
        
        return Up
    
    def loop(self, x, p, Psi, K, b):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 1d array, initial states
                   K -> float, kicks strength
                   b -> float, pseudo-impulsion
           Output : array, returns a state after one single iteration """
        
        L = len(x)
        res = Psi
        Up = self.Uprop(p,b)
         
        ## RECURSIVE DEFINITION TO AVOID DOUBLE LOOP ON THE BETA AVERAGE CALCULATIONS
        
        if nkick == 0:
            return res
        
        elif nkick == 1:
            fk = np.zeros(2*L, dtype=complex)
            fk[:L] = fft.fftshift(fft.fft(fft.ifftshift(res[:L])))
            fk[L:] = fft.fftshift(fft.fft(fft.ifftshift(res[L:])))
            Uk_f = np.matmul(Uk, fk)
            
            fp = np.zeros(2*L, dtype=complex)
            fp[:L] = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f[:L])))
            fp[L:] = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f[L:])))
            res = np.matmul(Up, fp)
            return res
        
        else:
            nkick = nkick - 1
            fk = np.zeros(2*L, dtype=complex)
            fk[:L] = fft.fftshift(fft.fft(fft.ifftshift(res[:L])))
            fk[L:] = fft.fftshift(fft.fft(fft.ifftshift(res[L:])))
            Uk_f = np.matmul(Uk, fk)
            
            fp = np.zeros(2*L, dtype=complex)
            fp[:L] = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f[:L])))
            fp[L:] = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f[L:])))
            res = np.matmul(Up, fp)
            return self.loop(x, p, res, Uk, K, nkick, b)            
    
    def avgPsi(self, x, p, Psi, K, t, n_avg):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 1d array, initial states
                   K -> float, kicks strength
                   t -> int, number of iterations of the simulation
                   n_avg -> int, number of beta (and omega) values to average on
           Output : array, returns the average density of probability for n_beta values of the pseudo-impulsion"""
        
        L = len(x)
        
        average = np.zeros((L,navg))
        Uk = self.Ukick(x,K,t)
        
        for i in range(navg):
            b = np.random.uniform(low=-0.5, high=0.5)
            psi_final = self.loop(x, p, Psi, Uk, K, nkick, b)
            average[:,i] = abs(psi_final[:L])**2 + abs(psi_final[L:])**2
            print(f'loop ended navg = {i}')
        
        return np.average(average, axis=1)
