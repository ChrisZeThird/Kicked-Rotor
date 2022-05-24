# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:04:49 2022

@author: ChrisZeThird
"""
"""
Two classes are defined here based on two different Hamiltoniens. The goal was to showcase an anti-CBS curve. The reason why a second Hamiltonien was defined
is because the first one wouldn't let me see the anti-CBS. 
"""

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

class SpinKickedRotor():
    
    def __init__(self,kb = 2.89):   
        self.I  = complex(0,1)
        self.kb = kb
        
    def Ukick(self, x, Psi, K):
        """Input : p -> array, impulsions
                   Psi -> 2d array, initial state
                   K -> float, kick's strength
           Output : array, returns the state after a Kick"""
        
        Uk = np.array([[np.cos(x) + np.sin(x),np.sin(2*x)],[np.sin(2*x),np.cos(x) - np.sin(x)]])
        Uk = np.exp(-(self.I*K/self.kb)*Uk)
        
        f1 = fft.fftshift(fft.fft(fft.ifftshift(Psi[0])))
        f2 = fft.fftshift(fft.fft(fft.ifftshift(Psi[1])))
        
        res_kick = np.array([Uk[0,0]*f1 + Uk[0,1]*f2, Uk[1,0]*f1 + Uk[1,1]*f2])
        
        return res_kick
        
    def Uprop(self, x, Psi, b=0):
        """Input : p -> array, impulsions
                   Psi -> 2d array, initial state
                   b -> float, pseudo-impulsion
           Output : array, returns the state after a Propagation"""
           
        zeros = np.zeros(len(x))
        
        Up = np.array([[(x + self.kb*b)**2,zeros],[zeros,(x + self.kb*b)**2]],dtype=complex)
        Up = np.exp(-(self.I*self.kb/2) * Up)
        
        f1 = fft.ifftshift(fft.ifft(fft.fftshift(Psi[0])))
        f2 = fft.ifftshift(fft.ifft(fft.fftshift(Psi[1])))
        
        res_prop = np.array([Up[0,0]*f1 + Up[0,1]*f2, Up[1,0]*f1 + Up[1,1]*f2])
        return res_prop    
    
    def loop(self, x, p, Psi, K, b, n):
    
        res = Psi
        
        for i in range(n):
                
            Uk_f = self.Ukick(x,res,K)
            
            norm = np.linalg.norm(Uk_f)
            res = self.Uprop(p,Uk_f/norm,b)
            
            norm = np.linalg.norm(res)
            
        return res/norm
    
    def avgPsi(self, x, p, Psi, K, t, n_beta):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   f -> array, initial state
                   K -> float, kicks strength
                   t -> int, number of iterations of the simulation
                   n_beta -> int, number of beta values to average on
           Output : array, returns the average density of probability for n_beta values of the pseudo-impulsion"""
        
        N = len(x)
        
        psi_final = np.zeros(N)
        
        Beta = np.random.uniform(low=-0.5, high=0.5, size=(n_beta,)) 
        
        for b in Beta:
            
            temp = self.loop(x,p,Psi,K,b,t)
            norm = np.linalg.norm(temp)
            temp /= norm
            
            psi_final += abs(temp[0])**2 + abs(temp[1])**2
        
        return psi_final/len(Beta)
        
    
class SpinKickedRotorRA():
    
    def __init__(self, omega, alpha, epsilon, kb = 2.89):   
        self.I  = complex(0,1)
        self.kb = kb
        self.omega = omega
        self.alpha = alpha
        self.epsilon = epsilon
        
    def Ukick(self, x, Psi, K, n_kick):
        """Input : p -> array, impulsions
                   Psi -> 2d array, initial state
                   K -> float, kick's strength
                   n_kick -> int, number of kicks
           Output : array, returns the state after a Kick"""
        
        zeros = np.zeros(len(x))
        
        Uk = K*(1 - ((-1)**n_kick)*self.alpha)*np.array([[np.cos(x) + self.epsilon*np.sin(x) + np.pi/2, zeros],[zeros, np.cos(x) - self.epsilon*np.sin(x) - np.pi/2]])
        Uk = np.exp(-(self.I/self.kb)*Uk)
        
        f1 = fft.fftshift(fft.fft(fft.ifftshift(Psi[0])))
        f2 = fft.fftshift(fft.fft(fft.ifftshift(Psi[1])))
        
        res_kick = np.array([Uk[0,0]*f1 + Uk[0,1]*f2, Uk[1,0]*f1 + Uk[1,1]*f2])
        norm = np.linalg.norm(res_kick)
        return res_kick/norm
        
    def Uprop(self, p, Psi, b=0):
        """Input : p -> array, impulsions
                   Psi -> 2d array, initial state
                   b -> float, pseudo-impulsion
           Output : array, returns the state after a Propagation"""
        
        ones = np.ones(len(p))
        
        Up = np.array([[(p + self.kb*b)**2,ones*self.omega],[ones*self.omega,(p + self.kb*b)**2]],dtype=complex)
        Up = np.exp(-(self.I/2/self.kb) * Up)
        
        f1 = fft.ifftshift(fft.ifft(fft.fftshift(Psi[0])))
        f2 = fft.ifftshift(fft.ifft(fft.fftshift(Psi[1])))
        
        res_prop = np.array([Up[0,0]*f1 + Up[0,1]*f2, Up[1,0]*f1 + Up[1,1]*f2])
        norm = np.linalg.norm(res_prop)
        return res_prop/norm
    
    def loop(self, x, p, Psi, K, b, n):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 2d array, initial states
                   K -> float, kicks strength
                   b -> float, pseudo-impulsion
                   n -> int, number of beta values to average on
           Output : array, returns the new state of the system after one kick and one propagations for n iterations"""
        
        res = Psi
        
        for i in range(n):
                
            Uk_f = self.Ukick(x,res,K,i+1)
            
            norm = np.linalg.norm(Uk_f)
            res = self.Uprop(p,Uk_f/norm,b)
            
            norm = np.linalg.norm(res)
            res = res/norm
            
        return res    
            
    
    def avgPsi(self, x, p, Psi, K, t, n_beta):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 2d array, initial states
                   K -> float, kicks strength
                   t -> int, number of iterations of the simulation
                   n_beta -> int, number of beta values to average on
           Output : array, returns the average density of probability for n_beta values of the pseudo-impulsion"""
        
        N = len(x)
        psi_init  = Psi
        psi_final = np.zeros(N)
        
        Beta = np.random.uniform(low=-0.5, high=0.5, size=(n_beta,)) 
        average = np.zeros((n_beta,N))
        
        j = 0
        
        for b in Beta:
            
            temp = self.loop(x,p,psi_init,K,b,t)
            
            average[j] = abs(temp[0])**2 + abs(temp[1])**2
            j += 1
            
        psi_final = np.average(average, axis=0)    
        return psi_final
   
        
        
