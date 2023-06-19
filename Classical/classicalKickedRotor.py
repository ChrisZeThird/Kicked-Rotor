# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:45:03 2022

@author: ChrisZeThird
"""
import numpy as np
import numpy.fft as npfft


class ClassicalKickedRotor:
    
    def __init__(self):
        pass

    @staticmethod
    def Chirikov(p, q, K, mod=True):
        """Input : p -> array, contains the impulsion
                   q -> array, contains the positions
                   K -> float, kicks strength
                   mod -> bool, if True, the value will then be %2π 
           Output : array, returns Chirikov standard map"""
        if mod:
            pn = (p + K*np.sin(q)) % (2*np.pi)
            qn = (q + pn) % (2*np.pi)
            return pn,qn
        else:
            pn = (p + K*np.sin(q)) 
            qn = (q + pn)
            return qn, pn
    
    def iteration(self, q0, p0, K, N, n):
        """Input : q0 -> float, initial positions
                   p0 -> float, initial impulsions
                   (q0 and p0 should be of same size)
                   N -> int, width of the studied space
                   K -> float, kicks strength
                   n -> number of iterations of the Chirikov standard map
           Output : tuple of arrays, Returns n iterations of the Chirikov stdmap with parameter K and resolution r
           
           This code is a slight rewriting of @t-makaro's version very useful to plot the phase graph. The ploting part isn't available in this file.
           """
        if len(p0) == len(q0):
             
            P, Q = np.meshgrid(p0, q0)
            Q = Q.reshape(len(p0)*len(q0))
            P = P.reshape(len(p0)*len(q0))  # resize phase space to 2 pi by 2 pi
            
            colours = np.array([Q/max(Q), (P+Q)/max(P+Q), P/max(P)]).T  # RGB value normalized
            print('colours', colours)
            Qn = [Q]
            Pn = [P]
            
            for i in range(n):
                Q, P = self.Chirikov(P, Q, K)
                Qn.append(Q)
                Pn.append(P)  
                
            print('h-stackQ', np.hstack(Qn))
            print('v-stackQ', np.vstack([colours]*(n+1)))
            return np.hstack(Qn), np.hstack(Pn), np.vstack([colours]*(n+1))  # Stack arrays in sequence horizontally (column wise).
        
        else:
            print('Initial conditions of positions and impulsion should have same size')

    def evolution(self, n, p0, q0, K):
        """Input : n -> int, width of the studied space
                    p0 -> float, initial impulsion
                    q0 -> float, initial position
                    K -> float, kicks strength
            Output : tuple of arrays, returns the evolution of the system from an initial state using Chirikov standard map equations"""
        
        p = np.linspace(0, 2*np.pi, n)
        q = np.linspace(0, 2*np.pi, n)
        p[0], q[0] = p0, q0
        
        for i in range(n-1):
            p[i+1], q[i+1] = self.Chirikov(p[i], q[i], K)
        return p,q
    
    def classicalEnergy(self, n, p0, q0, K):
        """Input : n -> int, width of the studied space
                   p0 -> array, initial impulsions
                   q0 -> array, initial positions
                   K -> float, kicks strength
           Output : array, returns the average value of the Kinetic energy in the classical approximation for a certain number of initial
                    conditions q0 and p0. The Kinetic energy corresponds to the value of <p²>."""
        
        P2 = np.zeros(n)
        
        for i in range(len(p0)):
            for j in range(len(q0)):
                p, _ = self.evolution(n, p0[i], q0[j], K)  # only p is useful in this case
                P2 += p**2
                
        return P2/n   
