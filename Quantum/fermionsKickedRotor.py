# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:55:13 2022
@author: ChrisZeThird
"""

import numpy as np
import numpy.fft as npfft
import matplotlib.pyplot as plt

""" As the kicked rotor was studied for only one particle distribution, I also worked on the case of multiple
    fermions distributions. Naturally, the functions from other file in this folder must be used if you want to
    see what happens in that very specific case. The use of fermions simply allows us to set one particle for each
    energy/impulsion."""

## Initial State 

def fermi(N,Nf):
    P = np.zeros((2*N, Nf), int)
    width = np.arange(-(Nf-1)/2,(Nf-1)/2 + 1, dtype=int) + N
    for i in range(Nf):
        P[width[i],i] = 1
    return P  

## Plot for Fermi distribution 

def fplot(farr):
    """" farr is the result of the function fermi above """
    N, Nf = np.shape(farr)
    # print('N= ', N)
    # print('Nf= ', Nf)
    plt.xlabel('p')
    plt.ylabel('Ψf')
    plt.title(f'Fermi distribution for {Nf} fermions and size {N}')
    plt.plot(np.linspace(-N/2,N/2-1,N),np.diag(np.dot(farr,np.transpose(np.conjugate(farr)))).real)
    plt.show()
