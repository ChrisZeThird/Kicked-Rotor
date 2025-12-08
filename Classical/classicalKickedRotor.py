# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:45:03 2022

@author: ChrisZeThird
"""
from numpy import pi, sin, linspace, zeros, meshgrid, array, hstack, vstack, ndarray
from typing import Tuple


class ClassicalKickedRotor:
    
    def __init__(self):
        pass

    @staticmethod
    def Chirikov(p: ndarray, q: ndarray, K: float) -> Tuple[ndarray, ndarray]:
        """Input : p -> array, contains the impulsion
                   q -> array, contains the positions
                   K -> float, kicks strength
                   mod -> bool, if True, the value will then be %2π 
           Output : array, returns Chirikov standard map"""
        pn = (p + K * sin(q)) % (2 * pi)
        qn = (p + q + K * sin(q)) % (2 * pi)

        return pn, qn
    
    def iteration(self, q0: ndarray, p0: ndarray, K: float, n: int) -> Tuple[ndarray, ndarray, ndarray]:
        """Input : q0 -> array, initial positions
                   p0 -> array, initial impulsions
                   (q0 and p0 should be of same size)
                   K -> float, kicks strength
                   n -> number of iterations of the Chirikov standard map
           Output : tuple of arrays, Returns n iterations of the Chirikov standard map with parameter K and resolution r
           
           This code is a slight rewriting of @t-makaro's version very useful to plot the phase graph. The plotting part
           isn't available in this file.
           """
        if len(p0) == len(q0):
             
            P, Q = meshgrid(p0, q0)

            Q = Q.reshape(len(p0) * len(q0))
            P = P.reshape(len(p0) * len(q0))  # resize phase space to 2 pi by 2 pi
            
            colours = array([Q / max(Q), (P + Q) / max(P + Q), P / max(P)]).T  # RGB value normalized

            Qn = [Q]
            Pn = [P]
            
            for i in range(n):
                P, Q = self.Chirikov(P, Q, K=K)
                Qn.append(Q)
                Pn.append(P)  

            return hstack(Pn), hstack(Qn), vstack([colours] * (n + 1))  # Stack arrays in sequence horizontally (column wise).
        
        else:
            print('Initial conditions of positions and impulsion should have same size')

    def evolution(self, n, p0, q0, K):
        """Input : n -> int, width of the studied space
                    p0 -> float, initial impulsion
                    q0 -> float, initial position
                    K -> float, kicks strength
            Output : tuple of arrays, returns the evolution of the system from an initial state using Chirikov standard map equations"""
        
        p = linspace(start=0, stop=2 * pi, num=n)
        q = linspace(start=0, stop=2 * pi, num=n)
        p[0], q[0] = p0, q0
        
        for i in range(n-1):
            p[i+1], q[i+1] = self.Chirikov(p[i], q[i], K)
        return p, q
    
    def classicalEnergy(self, n: int, p0: ndarray, q0: ndarray, K: float) -> ndarray:
        """Input : n -> int, width of the studied space
                   p0 -> array, initial impulsions
                   q0 -> array, initial positions
                   K -> float, kicks strength
           Output : array, returns the average value of the Kinetic energy in the classical approximation for a certain number of initial
                    conditions q0 and p0. The Kinetic energy corresponds to the value of <p²>."""
        
        P2 = zeros(n)
        
        for i in range(len(p0)):
            for j in range(len(q0)):
                p, _ = self.evolution(n, p0[i], q0[j], K)  # only p is useful in this case
                P2 += p ** 2
                
        return P2/n   
