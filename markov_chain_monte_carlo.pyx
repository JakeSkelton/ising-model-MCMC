# -*- coding: utf-8 -*-
"""
markov_chain_monte_carlo.pyx

Code intended as a library with two functions - metrohaste and metrohaste_stats
- to be compiled by Cython.
"""

import numpy as np
cimport cython
from libc.math cimport exp

@cython.boundscheck(False)
@cython.nonecheck(False)
def metrohaste(int numsteps, int[:,:] s, double H, double T, RNG):
    """
    A fast, Cython-based method to carry out one timestep of the Metropolis-
    -Hastings algorithm, a Markov chain Monte Carlo method. The lattice
    is stepped through in a left-to-right scanning fashion.

    Parameters
    ----------
    numsteps : int
        The number of Metropolis-Hastings timesteps to perform.
    s : square numpy array of ints
        The spin lattice, must be square and of type int.
    H : float
        The applied magnetic field, with dimensions (energy)/(spin).
    T : float
        The heat bath temperature, with dimensions of (energy).
    RNG : numpy.random generator object
        Seeded random number generator. Passing this to the function allows
        results to be repeated starting from a mother seed.

    Returns
    -------
    s_new : numpy array
        The updated spin lattice, ater 'numsteps' timesteps.

    """

    cdef Py_ssize_t N_ind = s.shape[0]
    cdef int N = s.shape[0]
    
    cdef double deltaE
    cdef double[:,:,:] p = RNG.uniform(0, 1, size=(numsteps,N,N))
    cdef Py_ssize_t t, i, j
    
    for t in range(numsteps):
        for i in range(N_ind):
            for j in range(N_ind):
                deltaE = (s[(i-1)%N, j] + s[(i+1)%N, j] +
                          s[i, (j-1)%N] + s[i, (j+1)%N] + 2*H)*s[i,j]
                s[i,j] *= (-1)**(exp(-deltaE/T) > p[t,i,j])
        
    return np.asarray(s)

@cython.boundscheck(False)
@cython.nonecheck(False)
def metrohaste_stats(int numsteps, int[:,:] s, double H, double T, RNG):
    """
    A fast, Cython-based method to carry out one timestep of the Metropolis-
    -Hastings algorithm, a Markov chain Monte Carlo method. The lattice
    is stepped through in a left-to-right scanning fashion. This version
    outputs statistics of the lattice over time, and is designed to called
    to output the entire length of a simulation [in practice this means it
    has to generate random numbers timestep-by-timestep, to conserve memory].

    Parameters
    ----------
    numsteps : int
        The number of Metropolis-Hastings timesteps to perform.
    s : square numpy array of ints
        The spin lattice, must be square and of type int.
    H : float
        The applied magnetic field, with dimensions (energy)/(spin).
    T : float
        The heat bath temperature, with dimensions of (energy).
    RNG : numpy.random generator object
        Seeded random number generator. Passing this to the function allows
        results to be repeated starting from a mother seed.

    Returns
    -------
    s_new : numpy array
        The updated spin lattice, ater 'numsteps' timesteps.
    sbars : numpy array    
        A 1D array of the lattice's mean spin at each timestep.
    energies : 
        A 1D array of the lattice's total energy at each timestep.
    """
    cdef Py_ssize_t N_ind = s.shape[0]
    cdef int N = s.shape[0]
    cdef double deltaE, E_t = 0
    cdef int count = 0, sqcount = 0
    cdef double[:,:] p = np.zeros((N, N))
    cdef double[:] sbars = np.zeros(numsteps, dtype=float)
    cdef double[:] energy = np.zeros(numsteps, dtype=float)
    cdef Py_ssize_t t, i, j
    
    for i in range(N_ind):
        for j in range(N_ind):
            E_t -= (0.25*(s[(i-1)%N, j] + s[(i+1)%N, j] +
                    s[i, (j-1)%N] + s[i, (j+1)%N]) + 2*H)*s[i,j]
    energy[0] = E_t
    
    for t in range(numsteps):
        count = 0
        sqcount = 0
        p = RNG.uniform(0, 1, size=(N,N))
        for i in range(N_ind):
            for j in range(N_ind):
                deltaE = (s[(i-1)%N, j] + s[(i+1)%N, j] +
                          s[i, (j-1)%N] + s[i, (j+1)%N] + 2*H)*s[i,j]
                if (exp(-deltaE/T) > p[i,j]):
                    s[i,j] *= -1
                    E_t += deltaE
                count += s[i,j]
                sqcount += s[i,j]*s[i,j]
        sbars[t] = <double>count/<double>(N*N)
        energy[t] = E_t
        
    return (np.asarray(s), np.asarray(sbars), np.asarray(energy))

@cython.boundscheck(False)
@cython.nonecheck(False)
def metrohaste_vect(int numsteps, int[:,:] s, double[:] H, double[:] T, RNG):
    """
    A fast, Cython-based method to carry out one timestep of the Metropolis-
    -Hastings algorithm, a Markov chain Monte Carlo method. The lattice
    is stepped through in a left-to-right scanning fashion. This version
    outputs statistics of the lattice over time, like metrohaste_stats, but
    takes temperature and magnetic field arguments as vectors of length
    'numsteps', in order to cycle the magnetic field or anneal the lattice
    according to a predefined trajectory.

    Parameters
    ----------
    numsteps : int
        The number of Metropolis-Hastings timesteps to perform.
    s : square numpy array of ints
        The spin lattice, must be square and of type int.
    H : float
        The applied magnetic field, with dimensions (energy)/(spin).
    T : float
        The heat bath temperature, with dimensions of (energy).
    RNG : numpy.random generator object
        Seeded random number generator. Passing this to the function allows
        results to be repeated starting from a mother seed.

    Returns
    -------
    s_new : numpy array
        The updated spin lattice, ater 'numsteps' timesteps.
    sbars : numpy array    
        A 1D array of the lattice's mean spin at each timestep.
    energies : 
        A 1D array of the lattice's total energy at each timestep.
    """
    cdef Py_ssize_t N_ind = s.shape[0]
    cdef int N = s.shape[0]
    cdef double deltaE, E_t = 0, T_t = T[0], H_t = T[0]
    cdef int count = 0, sqcount = 0
    cdef double[:,:] p = np.zeros((N, N))
    cdef double[:] sbars = np.zeros(numsteps, dtype=float)
    cdef double[:] energy = np.zeros(numsteps, dtype=float)
    cdef Py_ssize_t t, i, j
    
    for i in range(N_ind):
        for j in range(N_ind):
            E_t -= (0.25*(s[(i-1)%N, j] + s[(i+1)%N, j] +
                    s[i, (j-1)%N] + s[i, (j+1)%N]) + 2*H_t)*s[i,j]
    energy[0] = E_t
    
    for t in range(numsteps):
        T_t = T[t]
        H_t = H[t]
        count = 0
        sqcount = 0
        p = RNG.uniform(0, 1, size=(N,N))
        for i in range(N_ind):
            for j in range(N_ind):
                deltaE = (s[(i-1)%N, j] + s[(i+1)%N, j] +
                          s[i, (j-1)%N] + s[i, (j+1)%N] + 2*H_t)*s[i,j]
                if (exp(-deltaE/T_t) > p[i,j]):
                    s[i,j] *= -1
                    E_t += deltaE
                count += s[i,j]
                sqcount += s[i,j]*s[i,j]
        sbars[t] = <double>count/<double>(N*N)
        energy[t] = E_t
        
    return (np.asarray(s), np.asarray(sbars), np.asarray(energy))

            