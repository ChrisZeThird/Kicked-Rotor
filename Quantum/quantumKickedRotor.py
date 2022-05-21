# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:45:10 2022
@author: ChrisZeThird
"""

import numpy as np
import numpy.fft as npfft

class QuantumKickedRotor():
    
    def __init__(self,):
        self.kb = 2.89
        
    ## Utils 
    
    def squareModule(self, z):
        """Input : z -> array of complex
           Output : array, return the square module of a complex array of numbers"""
        r = np.real(z)
        i = np.imag(z)
        return r**2 + i**2
    
    def squareDot(self, x, arr):
        """Input : p -> array
                   arr -> array
           Output : array, return the dot products of x² and a given array"""
        return np.dot(x**2, arr)
    
    ## Fourier Transform 
    
    def FFT(self, f):
        """Input : array, f
           Output : array, computes the Fast Fourier Transform of an array"""
        return npfft.fft(f)
    
    def iFFT(self, f):
        """Input : f -> array
           Output : array, computes the Invert Fast Fourier Transform of an array"""
        return npfft.ifft(f)
    
    ## Operator
    
    def Ukick(self, x,f,K):
        """Input : x -> array, positions 
                   f -> array, initial state
                   K -> float, kicks strength
           Output : array, computes the Pulse Operator of the kicked rotor given f an initial state"""
        Uk = np.diag(np.exp(-1j*K*np.cos(x)/self.kb))
        return np.dot(Uk, npfft.fftshift(self.FFT(npfft.ifftshift(f))))
        
    def Uprop(self, x,f,b=0):
        """Input : x -> array, positions 
                   f -> array, initial state
                   b -> float, pseudo-impulsion in [0,1]
           Output : array, computes the Propagation Operator of the kicked rotor given f a state"""
        Up = np.diag(np.exp(-1j*(x + self.kb*b)**2/2/self.kb))
        return np.dot(Up, npfft.fftshift(self.iFFT(npfft.ifftshift(f))))
    
    ## Sequence 
    
    def loop(self, x,p,f,K,b=0):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   f -> array, initial state
                   K -> float, kicks strength
                   b -> float, pseudo-impulsion in [0,1]
           Output : array, numerical simulation of the kicked rotor, based on a Fast Fourier Transform algorithm, returns the final
                    distribution after one iteration"""

        Uk_f = self.Ukick(x,f,K)
        Up_f = self.Uprop(p,Uk_f,b)
            
        return Up_f
    
    ## Average values of Energy and Probability density
    
    def avgPsi(self, x, p, f, K, t, n_beta):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   f -> array, initial state
                   K -> float, kicks strength
                   t -> int, number of iterations of the simulation
                   n_beta -> int, number of beta values to average on
           Output : array, return the average density of probability for n_beta values of the pseudo-impulsion"""
           
        Beta = np.random.uniform(low=-0.5, high=0.5, size=(n_beta,))    
           
        avgPsi = np.zeros(len(p)) 
        for b in Beta:
            res = f
            res = self.loop(x,p,res,K,b)        
            avgPsi += self.squareModule(res)
           
        return avgPsi/n_beta
                   
    def avgEnergy(self, x, p, f, K, n, n_beta):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   f -> array, initial state
                   K -> float, kicks strength
                   n -> int, number of iterations of the simulation
                   n_beta -> int, number of beta values to average on
           Output : array, return the average energy for n_beta values of the pseudo-impulsion"""
                    
        Beta = np.random.uniform(low=-0.5, high=0.5, size=(n_beta,))    
           
        avgE = np.zeros(n)
        res = np.zeros(n)
        
        init = f
        
        for b in Beta:
            f = init
            for i in range(n):
                        
                f = self.loop(x,p,f,K,b)        
                res[i] = self.squareDot(p,self.squareModule(f))
                        
            avgE += res
                
        avgE /= n_beta 
        return avgE*(self.kb**2/2)
