# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:04:49 2022

@author: ChrisZeThird
"""
"""
Two classes are defined here based on two different Hamiltoniens. The goal was to showcase an anti-CBS curve. The reason why a second Hamiltonien was defined
is because the first one wouldn't let me see the anti-CBS. But the reason was the missunderstanding of the operation. The function np.exp of numpy doesn't compute
the exponential of a matrix but the exponential of each elements of the array. For that reason, I had to redefine the funcion, using the scipy module function expm
which calculates the exponential of the matrix. The first model was then successful
"""

import numpy as np
from scipy.linalg import expm
import numpy.fft as fft
import matplotlib.pyplot as plt

import time

class SpinKickedRotor():
    
    def __init__(self, mu, epsilon, kb = 2.89):   
        self.I  = complex(0,1)
        self.kb = kb
        self.mu = mu
        self.epsilon = epsilon
        
    def Ukick(self, x, K):
        """Input : x -> array, impulsions
                   K -> float, kick's strength
           Output : array, returns the state after a Kick"""
        
        L = len(x)
        
        x_diag = np.diag(x)
        
        # fk = fft.fftshift(fft.fft(fft.ifftshift(Psi)))
        
        Uk = np.zeros((2*L,2*L))
        
        U1 = np.cos(x_diag) + self.epsilon * np.sin(x_diag)
        U2 = 1/2 * self.mu * np.sin(2*x_diag)
        U3 = 1/2 * self.mu *np.sin(2*x_diag)
        U4 = np.cos(x_diag) - self.epsilon * np.sin(x_diag)
        
        Uk[:L,:L]       = U1
        Uk[:L,L:2*L]    = U2
        Uk[L:2*L,:L]    = U3
        Uk[L:2*L,L:2*L] = U4
        
        Uk = expm(-(self.I*K/self.kb)*Uk)
                
        return Uk
        
    def Uprop(self, p,b=0):
        """Input : p -> array, impulsions
                   b -> float, pseudo-impulsion
           Output : array, returns the state after a Propagation"""
           
        L = len(p)
        
        p_diag = np.diag(p)
        zeros = np.zeros((L,L))
        
        #fp = fft.ifftshift(fft.ifft(fft.fftshift(Psi)))
        
        Up = np.zeros((2*L,2*L))
        
        U1 = ((p_diag + b)**2)
        U2 = zeros
        U3 = zeros
        U4 = ((p_diag + b)**2)
        
        Up[:L,:L]       = U1
        Up[:L,L:2*L]    = U2
        Up[L:2*L,:L]    = U3
        Up[L:2*L,L:2*L] = U4

        Up = expm(-(self.I*self.kb/2) * Up)
        
        # f1 = fft.ifftshift(fft.ifft(fft.fftshift(Psi[0])))
        # f2 = fft.ifftshift(fft.ifft(fft.fftshift(Psi[1])))
        
        # res_prop = np.matmul(Up,fp)
        return Up    
    
    def loop(self, x, p, Psi, K, b, n):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 1d array, initial states
                   K -> float, kicks strength
                   b -> float, pseudo-impulsion
           Output : array, returns a state after one single iteration """
        
        res = Psi
        
        Uk = self.Ukick(x,K)
        Up = self.Uprop(p,b)
         
        fk = fft.fftshift(fft.fft(fft.ifftshift(res)))
        Uk_f = np.matmul(Uk, fk)
        # print(Uk_f.shape)
        fp = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f)))
        res = np.matmul(Up, fp)
             
        # norm = np.linalg.norm(res)    
        return res
    
    def avgPsi(self, x, p, Psi, K, t, n_avg):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 1d array, initial states
                   K -> float, kicks strength
                   t -> int, number of iterations of the simulation
                   n_avg -> int, number of beta (and omega) values to average on
           Output : array, returns the average density of probability for n_beta values of the pseudo-impulsion"""
        
        L = len(x)
        psi_final = np.zeros(L)
        
        Beta = np.random.uniform(low=-0.5, high=0.5, size=(n_avg,)) 

        average = np.zeros((n_avg,L))
        tick = time.time()
        Uk = self.Ukick(x,K)
        tock = time.time()
        # print('Uk', tock-tick)
        
        for i in range(n_avg):
            tick_i = time.time()
            b = Beta[i]
            Up = self.Uprop(p,b)
            res = Psi    
            
            for l in range(t):
                fk = fft.fftshift(fft.fft(fft.ifftshift(res)))
                Uk_f = np.matmul(Uk, fk)
                # print(Uk_f.shape)
                fp = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f)))
                res = np.matmul(Up, fp)
                # print(res.shape)
            # print(res[:L])
            # print(res[L:1*L])
            average[i,:] = abs(res[:L])**2 + abs(res[L:2*L])**2
            tock_i = time.time()
            print(f'loop {i} took {tock_i - tick_i}s')
            
        # print(average)
        # print(average.shape)
        psi_final = np.average(average, axis=0)    
        return psi_final
        
    
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
        x_diag = np.diag(x)
        
        Uk = np.zeros((2*L,2*L))
        
        U1 = np.cos(x_diag) + self.epsilon * np.sin(x_diag)
        U2 = zeros
        U3 = zeros
        U4 = np.cos(x_diag) - self.epsilon * np.sin(x_diag)
        
        # f1 = fft.fftshift(fft.fft(fft.ifftshift(Psi[0])))
        # f2 = fft.fftshift(fft.fft(fft.ifftshift(Psi[1])))
        
        # fk = fft.fftshift(fft.fft(fft.ifftshift(Psi)))
        
        coeff = K*(1 - ((-1)**n_kick) * self.alpha)
        
        Uk[:L,:L]       = U1
        Uk[:L,L:2*L]    = U2
        Uk[L:2*L,:L]    = U3
        Uk[L:2*L,L:2*L] = U4
        
        Uk = expm(-(self.I/self.kb)*coeff*Uk)
        
        # res_kick = np.dot(Uk,fk)
        # norm = np.linalg.norm(res_kick)
        return Uk
        
    def Uprop(self, p, b=0):
        """Input : p -> array, impulsions
                   Psi -> 2d array, initial state
                   b -> float, pseudo-impulsion
           Output : array, returns the state after a Propagation"""
        
        L = len(p)
        
        identity = np.identity(L)
        p_diag = np.diag(p)
        
        Up = np.zeros((2*L,2*L))
        
        U1 = (p_diag + self.kb*b)**2 
        U2 = identity*(self.omega)
        U3 = identity*(self.omega)
        U4 = (p_diag + self.kb*b)**2 
    
        Up[:L,:L]       = U1
        Up[:L,L:2*L]    = U2
        Up[L:2*L,:L]    = U3
        Up[L:2*L,L:2*L] = U4
        
        Up = expm(-(self.I/2/self.kb) * Up)
        
        # fp = fft.ifftshift(fft.ifft(fft.fftshift(Psi)))
        
        # f1 = fft.ifftshift(fft.ifft(fft.fftshift(Psi[0])))
        # f2 = fft.ifftshift(fft.ifft(fft.fftshift(Psi[1])))
        
        # res_prop = np.matmul(Up,fp)
        # norm = np.linalg.norm(res_prop)
        return Up
    
    def loop(self, x, p, Psi, K, b, n):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 1d array, initial states
                   K -> float, kicks strength
                   b -> float, pseudo-impulsion
           Output : array, returns a state after one single iteration """
        
        res = Psi
        
        Uk = self.Ukick(x,K)
        Up = self.Uprop(p,b)
         
        fk = fft.fftshift(fft.fft(fft.ifftshift(res)))
        Uk_f = np.matmul(Uk, fk)
        # print(Uk_f.shape)
        fp = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f)))
        res = np.matmul(Up, fp)
             
        # norm = np.linalg.norm(res)    
        return res            
    
    def avgPsi(self, x, p, Psi, K, t, n_avg):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 1d array, initial states
                   K -> float, kicks strength
                   t -> int, number of iterations of the simulation
                   n_avg -> int, number of beta (and omega) values to average on
           Output : array, returns the average density of probability for n_beta values of the pseudo-impulsion"""
        
        L = len(x)
        psi_init  = Psi
        psi_final = np.zeros(L)
        
        Beta = np.random.uniform(low=-0.5, high=0.5, size=(n_avg,)) 

        average = np.zeros((n_avg,L))
        tick = time.time()
        Uk = self.Ukick(x,K,t)
        tock = time.time()
        print('Uk', tock-tick)
        
        for i in range(n_avg):
            b = Beta[i]
            res = Psi     
            Up = self.Uprop(p,b)
            for time in range(t):
                fk = fft.fftshift(fft.fft(fft.ifftshift(res)))
                Uk_f = np.matmul(Uk, fk)
                fp = fft.ifftshift(fft.ifft(fft.fftshift(Uk_f)))
                res = np.matmul(Uk_f,fp)
            
            average[i] = abs(res[:L])**2 + abs(res[L:2*L])**2
            
        psi_final = np.average(average, axis=0)    
        return psi_final  
