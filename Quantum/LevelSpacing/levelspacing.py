# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:36:50 2022

@author: ChrisZeThird
"""
from numpy import pi, diff, zeros, ones, diag, linspace, random, linalg, sort, angle, exp, ravel, ndarray
from Quantum.quantumKickedRotor import KickedRotor as kr

sim = kr()


# Mean level spacing
def meanlvl(E: ndarray) -> float:
    """
    Input : E -> array, energy levels
    Output : float, mean level spacing
    """
    L = len(E)
    return (1 / (L - 1)) * sum(diff(E))


def lvlspacing(m: int, K: float, nbeta: int) -> ndarray:
    """
    Input : m -> int, size parameter (N=2^m)
            K -> float, kick strength
            nbeta -> int, number of samples
    Output : array, level spacing distribution
    """
    N = 2 ** m
    quasiENorm = zeros((2 * N - 1, nbeta))
    Psi = diag(ones(
        2 * N))  # Contains all possible states (first component is a dirac, the second one a ket of 0, and vice versa)
    p = linspace(start=-N, stop=N, endpoint=False, num=2 * N)  # impulsions
    x = pi / N * p  # positions

    for i in range(nbeta):
        b = random.uniform(low=-0.5, high=0.5)
        operator = zeros((2 * N, 2 * N),
                         dtype=complex)  # will contain all the final state for every possible initial distribution

        for v in range(2 * N):
            psi_final = sim.loop(x, p, Psi[:, v], K, 1,
                                 b)  # Depends on the simulation you're using, here it is for the orthogonal case
            operator[:, v] = psi_final

        val = linalg.eigvals(operator)
        phi_n = sort(
            angle(val) % (2 * pi))  # /!\ WARNING ADD [::2] IN THE SYMPLECTIC CASE BECAUSE OF THE KRAMER DEGENERACY
        D = diff(phi_n)
        avg = meanlvl(phi_n)

        quasiE = D / avg

        quasiENorm[:, i] = quasiE
        if i % 100 == 0:
            print(f'loop {i}')

    return ravel(quasiENorm)


# Level Spacing distribution (does not include the Unitary case)
def Poisson(s: float) -> float:
    """
    Input : s -> float, level spacing
    Output : float, Poisson distribution value at s
    """
    return exp(-s)


def Orthogonal(s: float) -> float:
    """
    Input : s -> float, level spacing
    Output : float, Orthogonal distribution value at s
    """
    return (s * pi / 2) * exp(-s ** 2 * pi / 4)


def Symplectic(s: float) -> float:
    """
    Input : s -> float, level spacing
    Output : float, Symplectic distribution value at s
    """
    return (s ** 4 * 2 ** 18 / 3 ** 6 / pi ** 3) * exp(-s ** 2 * 64 / 9 / pi)
