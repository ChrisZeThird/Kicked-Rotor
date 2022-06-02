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

class SpinKickedRotor():
    
    def __init__(self, mu, epsilon, kb = 2.89, optRKR=False):   
        self.I  = complex(0,1)
        self.kb = kb
        self.mu = mu
        self.epsilon = epsilon
        self.optRKR = optRKR
        
    def Ukick(self, x, K):
        """Input : x -> array, impulsions
                   K -> float, kick's strength
           Output : array, returns the state after a Kick"""
        
        k = K/self.kb
        
        L = len(x)
        
        Uk = np.zeros((2*L,2*L), dtype=complex)
        
        if (self.mu != 0) and (self.epsilon != 0) :
            s1 = np.sqrt(self.mu**2 * ((np.cos(x)))**2 + self.epsilon**2)
            s2 = k * (np.sin(x)) * s1
            
            U1 = np.exp(-(self.I*k)*np.cos(x)) * (np.cos(s2) - self.epsilon*self.I*np.sin(s2)/s1)
            U23 = np.exp(-(self.I*k)*np.cos(x)) * (-self.mu*self.I*np.cos(x)*np.sin(s2)/s1)
            U4 = np.exp(-(self.I*k)*np.cos(x)) * (np.cos(s2) + self.epsilon*self.I*np.sin(s2)/s1)
            
            Uk[:L,:L]       = np.diag(U1)
            Uk[:L,L:2*L]    = np.diag(U23)
            Uk[L:2*L,:L]    = np.diag(U23)
            Uk[L:2*L,L:2*L] = np.diag(U4)
        
        else: # takes into account the classical quantum kicked rotor case
            U1 = np.exp(-(self.I*k)*np.cos(x))
            
            Uk[:L,:L]       = np.diag(U1)
            Uk[L:2*L,L:2*L] = np.diag(U1)
        
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
    
    def loop(self, x, p, Psi, Uk, Up, K, nkick):
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
            
            return self.loop(x, p, res, Uk, Up, K, nkick)
    
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
            psi_final = self.loop(x, p, Psi, Uk, Up, K, nkick)
            average[:,i] = abs(psi_final[:L])**2 + abs(psi_final[L:])**2
            if i % 10 == 0:
                print(f'loop {i}')
        
        return np.average(average, axis=1)
    
    
    
class SpinKickedRotorRA():
    
    def __init__(self, alpha, epsilon=0, kb=2.89, optRKR=False):   
        self.I  = complex(0,1)
        self.kb = kb
        # self.omega = omega
        self.alpha = alpha
        self.epsilon = epsilon
        self.optRKR = optRKR
        
    def Ukick(self, x, K, nkick):
        """Input : x -> array, positions
                   K -> float, kick's strength
                   nkick -> int, number of kicks
           Output : array, returns the state after a Kick"""
        
        L = len(x)
        k = K/self.kb
        
        zeros = np.zeros((L,L), dtype=complex)
        Uk = np.zeros((2*L,2*L), dtype=complex)
        
        coeff = (1 + ((-1)**nkick)*self.alpha)
        rep1 = np.cos(x)
        # rep2 = (np.pi/2)*np.sign(x)
        rep2 = self.epsilon * np.sin(2*x) + np.pi/2
        U1 = np.exp((-self.I*k)*(rep1 + rep2)*coeff)
        U4 = np.exp((-self.I*k)*(rep1 - rep2)*coeff)
        
        Uk[:L,:L]       = np.diag(U1)
        Uk[:L,L:2*L]    = zeros
        Uk[L:2*L,:L]    = zeros
        Uk[L:2*L,L:2*L] = np.diag(U4)
        
        return Uk
        
    def Uprop(self, p, b=0, omega=0):
        """Input : p -> array, impulsions
                   Psi -> 2d array, initial state
                   b -> float, pseudo-impulsion
                   omega -> float
           Output : array, returns the state after a Propagation"""
        
        L = len(p)
        
        Up = np.zeros((2*L,2*L), dtype=complex)
        
        if self.optRKR:
            phiVect = 2*np.pi*np.random.rand(L)
            P =  np.diag(np.exp(-self.I*phiVect))
        else:
            P =  np.diag(np.exp(-(self.I*self.kb/2) * ((p + b)**2)))
        
        U14 = P*np.cos(omega/2/self.kb)
        U23 = P*(-self.I*np.sin(omega/2/self.kb))
        

        Up[:L,:L]       = U14
        Up[:L,L:2*L]    = U23
        Up[L:2*L,:L]    = U23
        Up[L:2*L,L:2*L] = U14

        return Up
    
    def loop(self, x, p, Psi, K, Up, nkick):
        """Input : x -> array, positions 
                   p -> array, impulsions
                   Psi -> 1d array, initial states
                   K -> float, kicks strength
                   Up -> 2d array, propagation operator
                   nkick -> int, number of kicks
           Output : array, returns a state after one single iteration """
        
        res = Psi
        L = len(x)
        
        Uk = self.Ukick(x, K, nkick)
        
        # ## RECURSIVE DEFINITION TO AVOID DOUBLE LOOP ON THE BETA AVERAGE CALCULATIONS
        
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
            return self.loop(x, p, res, K, Up, 0)
        
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
            return self.loop(x, p, res, K, Up, nkick)
        
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
        # Uk = self.Ukick(x, K, nkick)
        
        for i in range(navg):
            b = np.random.uniform(low=-0.5, high=0.5)
            omega = np.random.uniform(low=-np.pi, high=np.pi)
            Up = self.Uprop(p,b,omega)
            psi_final = self.loop(x, p, Psi, K, Up, nkick)
            average[:,i] = abs(psi_final[:L])**2 + abs(psi_final[L:])**2
            if i % 50 == 0:
                print(f'loop {i}')
        
        return np.average(average, axis=1)
