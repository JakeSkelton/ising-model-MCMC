# -*- coding: utf-8 -*-
"""
comp_project_0_js2443.py

Main library for my computing project. Includes the 'lattice' class.
"""

import numpy as np
import markov_chain_monte_carlo as mcmc
import multiprocessing as multi
import time
from functools import partial
import matplotlib.pyplot as plt
import plotaesthetics

class lattice:
    """
    A simple class to wrap the important aspects of a lattice of Ising model
    spins - number of lattice sites, applied field, temperature etc. Also
    includes methods to step the lattice forward in time according to the
    Metropolis-Hastings algorithm and to calculate statistics.
    """
    def __init__(self, sidelength, magfield, temp, uniform=False,
                 seed=None):
        """
        Initialise the lattice. Stores the characteristics passed as
        arguments but also calculates the mean spin (mean magnetisation per 
        site) and variance in spin.

        Parameters
        ----------
        sidelength : int
            Square root of the number of lattice sites in the square lattice.
        magfield : float
            Applied magnetic field (permeability equals unity), with 
            dimensions (energy)/(spin).
        temp : float
            Heat bath temperature, with dimensions of (energy) (k_B := 1).
        uniform : bool, optional
            Whether to initialise the lattice with spins all pointing up.
            Otherwise, a random lattice is generated. The default is False.
        seed : int
            Seed for the random number generator assigned to this lattice, 
            which is used in the 'stepforward' method to evolve the lattice in
            time.

        Returns
        -------
        None.

        """
        self.time = 0
        self.N = sidelength
        self.H = magfield
        self.T = temp
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        if uniform == 'checkerboard':
            ## For testing purposes
            self.spins = (-1)**(np.sum(np.indices((self.N, self.N)),
                                       axis=0)%2)
        if uniform == 'dynamic':
            ## uniform for T < 1, random for T > 1
            if self.T < 1.0:
                self.spins = ((-1)**self.rng.integers(0,2)*
                              np.ones((self.N, self.N), dtype=int))
            else:
                self.spins = (-1)**self.rng.integers(0, 2,
                                         size=(self.N, self.N), dtype=int)
        elif uniform:
            self.spins = ((-1)**self.rng.integers(0,2)*
                          np.ones((self.N, self.N), dtype=int))
        else:
            self.spins = (-1)**self.rng.integers(0, 2, size=(self.N, self.N),
                                                 dtype=int)
        self.E = self.updateE()
            
    def __str__(self):
        """
        Create a toy string version of the spin lattice when the argument to
        'print'; prints a blank space where s = -1, and a unicode square where
        s = +1.
        """
        string = ''
        for i in range(self.N):
            for j in range(self.N):
                string += '\u25a0 ' if (self.spins[i,j] + 1) else '  '
            string += '\n'
        return string
    
    def stepforward(self, numsteps=1, desiredout='spins'):
        """
        A Cython-enhanced method to carry out the Metropolis-Hastings 
        algorithm on the spin lattice, for one timestep.
        This function is dependent on the 'metrohaste' import, which in turn 
        is dependent on a successfully compiled 
        'markov_chain_monte_carlo.pyx'. If a link in this chain does 
        not work, please use 'stepforward_slow', but there will be a 
        performance penalty of around 100x.

        Parameters
        ----------
        numsteps : int, optional
            Number of timesteps to move. Only the final spin lattice is
            returned for numsteps > 1. The default is 1.
        desiredout : string, optional
            One of 'stats' and 'spins'. If 'stats' is passed, the mean and
            variance of the spins at each timestep is returned; if 'spins' is
            passed, the lattice is skipped to its final state and the
            array of spins returned. The default is 'spins'.

        Returns
        -------
        Either the tuple ('sbars(t)', 'energies(t)', burn-in time) or the 
        final spin lattice after 'numsteps'.

        """
        s = self.spins
        if desiredout == 'stats':
            out = mcmc.metrohaste_stats(numsteps, s, self.H, self.T, self.rng)
            self.spins = out[0]
            self.E = out[3][-1]
            if self.time == 0:
                burn = self.findburnins_highT(out[1])
            else:
                burn = np.nan
            self.time += numsteps
            return (out[1], out[2], out[3], burn)
        else:
            self.spins = mcmc.metrohaste(numsteps, s, self.H, self.T, self.rng)
            self.time += numsteps
            return self.spins
       
    @staticmethod    
    def stepforward_l(latlist, numsteps=1, parallel=False):
        """
        As for 'stepforward', but now a list of lattices should be passed.
        This allows for vectorisation of statistical calculations over an 
        ensemble of Markov chains (burn-in times in particular).

        Parameters
        ----------
        latlist : iterable
            list, tuple or numpy array of lattice objects.
        numsteps : int, optional
            See 'stepforward'. The default is 1.

        Returns
        -------
        See 'stepforward. Now numpy arrays of the usual returns are 
        returned, with the first index corresponding to the particular
        Markov chain.

        """
        numchains = len(latlist)
        sbars = np.zeros((numchains, numsteps))
        Es = sbars.copy()
        if (latlist[0].N*numsteps > 100000 and numchains > 1) and parallel:
            print('Parallelism invoked')
            funcs = [partial(mcmc.metrohaste_stats, numsteps,
                             l.spins, l.H, l.T, l.rng) for l in latlist]
            outs = parallelise(funcs)
            for m in range(numchains):
                latt = latlist[m]
                out = outs[m]
                latt.spins = out[0]
                sbars[m], Es[m] = out[1:3]
                latt.E = Es[m,-1]
        else:
            for m in range(numchains):
                latt = latlist[m]
                out = mcmc.metrohaste_stats(
                    numsteps, latt.spins, latt.H, latt.T, latt.rng)
                latt.spins = out[0]
                sbars[m], Es[m] = out[1:3]
                latt.E = Es[m,-1]
        Ts = np.array([l.T for l in latlist])
        burns = lattice.findburnins(sbars, Ts, bound=1/(latlist[0].N))
        
        return(sbars, Es, burns)
    
    @staticmethod    
    def stepforward_vect(latlist, H, T, numsteps=1):
        """
        As for 'stepforward', but now a list of lattices should be passed.
        This allows for vectorisation of statistical calculations over an 
        ensemble of Markov chains (burn-in times in particular).

        Parameters
        ----------
        latlist : iterable
            list, tuple or numpy array of lattice objects.
        numsteps : int, optional
            See 'stepforward'. The default is 1.

        Returns
        -------
        See 'stepforward. Now numpy arrays of the usual returns are 
        returned, with the first index corresponding to the particular
        Markov chain.

        """
        numchains = len(latlist)
        sbars = np.zeros((numchains, numsteps))
        Es = sbars.copy()
        if latlist[0].N*numsteps > 100000 and numchains > 1:
            print('Parallelism invoked')
            funcs = [partial(mcmc.metrohaste_vect, numsteps,
                             latlist[m].spins, H[m,:], T[m,:], latlist[m].rng)
                             for m in range(numchains)]
            outs = parallelise(funcs)
            for m in range(numchains):
                latt = latlist[m]
                out = outs[m]
                latt.spins = out[0]
                sbars[m], Es[m] = out[1:3]
                latt.E = Es[m,-1]
        else:
            for m in range(numchains):
                latt = latlist[m]
                out = mcmc.metrohaste_vect(
                    numsteps, latt.spins, H[m,:], T[m,:], latt.rng)
                latt.spins = out[0]
                sbars[m], Es[m] = out[1:3]
                latt.E = Es[m,-1]
        burns = lattice.findburnins_highT(sbars, bound=1e-2/latlist[0].N)
        
        return(sbars, Es, burns)
    
    def updateE(self):
        """
        Manually re-compute the total energy of the lattice, using the
        array of spins, at the current time.
        """
        s = self.spins
        self.E = -np.sum((0.25*(np.roll(s, 1, axis=0) + np.roll(s, -1, axis=0)+ 
                                np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1))
                          + 2*self.H)*s)
        return self.E
    
    @staticmethod
    def findburnins(sbars, Ts, Tthresh=0.8, bound=1e-2, binsize=10):
        """
        A hybrid method to determine burn-in that takes an input of mean spins
        over time, 'sbars', and splits it according to the corresponding 
        lattice temperatures 'Ts'. Temperatures higher than 'bound' have their
        spins passed to 'findburnins_highT'. Lower temperatures are operated on
        by this function, which examines when the mean spin - split into time
        averaged chunks - gets closer to +/- 1 than bound.

        Parameters
        ----------
        sbars : numpy array
            m x n array of mean spin over n numsteps for m lattices.
        Ts : numpy array
            Array of length m containing corresponding lattice temperatures.
        Tthresh : float, optional
            The temperature boundary at which to assign the lattices to 
            different algorithms. The default is 0.8.
        bound : float, optional
            The user-chosen value that (|equilibrium spin| - 1) should not
            be expected to exceed. The default is 1e-2.
        binsize : int, optional
            The size of the chunks over which 'sbars' should be time-averaged.
            This helps mitigate the effect of thermal noise. The default is 10.

        Returns
        -------
        burns : numpy array
            Array with the same dimensions as 'Ts' - the burn-in time for each
            lattice.

        """
        
        try:
             numchains = sbars.shape[0]
             numsteps = sbars.shape[1]
        except IndexError as e:
            print(e)
            numchains = 1
            numsteps = sbars.shape[0]
            sbars = np.reshape(sbars, (1, numsteps))
            Ts = np.reshape(Ts, (1, 1))
        
        burns = (numsteps-1)*np.ones(numchains, dtype=int)
        lowTinds = np.arange(numchains, dtype=int)[Ts < Tthresh]
        highTinds = np.arange(numchains, dtype=int)[np.logical_not(Ts<Tthresh)]
        
        binmean = np.cumsum(sbars - 
                            np.hstack((np.zeros((numchains, binsize)),
                                       sbars[:,:-binsize])), axis=1)/binsize
        if highTinds.size:
            burns[highTinds] = lattice.findburnins_highT(sbars[highTinds,:],
                                                         bound=bound/100)
        chains_satis = np.any(np.abs(np.abs(binmean[lowTinds,:]) - 1) < bound,
                              axis=1)
        burnslow = burns[lowTinds]      ## Juggling array definitions to modiy
                                        ## arrays in place
        burnslow[chains_satis] = np.argmax(
            np.abs(np.abs(binmean[lowTinds,:]) - 1) < bound,
            axis=1)[chains_satis]
        burns[lowTinds] = burnslow

        return burns
    
    @staticmethod
    def findburnins_highT(sbars, bound=5e-4):
        """
        An alternative to method to determine the burn-in time that handles
        thermal fluctuations better but cannot diagnose low temperature 
        metastability. This method focuses on finding when the cumulative
        average of mean spins,
        $$ \frac{1}{t}\sum_{t'=0}^{t} \frac{\bar{s}(t')} $$
        falls below a user-defined 'bound'. This is observed to discriminate
        quite accurately between relatively short burn-ins.

        Parameters
        ----------
        sbars : numpy array
            Either a 1D array of mean spins at each timestep for a single
            lattice, or a 2D array with the first index denoting which lattice
            the mean spins correspond to.
        bound : float, optional
            The user-chosen value to which the average should relax before the
            label 'bured-in' applies. The default is 1e-3.

        Returns
        -------
        burns : int or numpy array of ints
            The determined burn-in time(s) of the supplied lattice(s).

        """
        try:
             numchains = sbars.shape[0]
             numsteps = sbars.shape[1]
        except IndexError as e:
            print(e)
            numchains = 1
            numsteps = sbars.shape[0]
            sbars = np.reshape(sbars, (1, numsteps))
            
        cmean = np.cumsum(sbars, axis=1)/np.arange(1, numsteps + 1)
        burns = np.argmin(np.abs(np.abs(np.diff(cmean, axis=1)) - bound),
                          axis=1)
        return burns
    
    
    @staticmethod
    def autocorrelation(sbars, burns):
        """
        Computes the autocorrelation of all the rows in 'sbars', discarding
        those values before burn-in. The different burn-ins for each row 
        means that a loop over the number of Markov chains has to be 
        performed, unforunately; np.correlate is used at each iteration,
        which is fast because it uses FFT behind the scenes.
        The output of this function is formatted so that if 'sbars' is a
        MxN array, 'autocorrs' will be a Mx(2N+1) array; this is quite 
        familiar for FFTs.

        Parameters
        ----------
        sbars : numpy array
            1D or 2D array of mean spins over time. Time should evolve along
            the second axis.
        burns : numpy array
            1D numpy array of the same length as sbars' first axis, containg 
            the burn-in times for each Markov chain.

        Returns
        -------
        autocorrs : numpy array
            Autocorrelation of each of the row vectors of 'sbars', with shape
            as described in the header. Cutting off the parts of each spin-
            -over-time vector pre burn-in leads to autocorrelation vectors of
            differing sizes (2*(numsteps-burnin)+1 to be specific), so each
            autocorrelation is symmetrically padded with zeros.
        decorrtimes : numpy array
            Array of decorrelation times (when autocorrelation first dips 
            below 1/e), of the same shape as 'burns'.
        """
        try:
            numsteps = sbars.shape[1]
            numchains = sbars.shape[0]
        except IndexError:
            numsteps = sbars.size
            numchains = 1
            sbars = sbars.reshape(1, numsteps)
            burns = burns.reshape(1)
    
        gsteps = numsteps - burns
        autocovs = np.zeros((numchains, 2*numsteps+1))
        for m in range(numchains):
            dev = sbars[m,burns[m]:] - np.mean(sbars[m,burns[m]:])
            out = np.correlate(dev, dev, mode='full')
            autocovs[m,:] = np.pad(out, (burns[m]+1,burns[m]+1),
                                   'constant', constant_values=(0,0))
            
        autocorrs = (autocovs.T/autocovs[:,numsteps]).T
        ## Finds first instance where |autocorrelation| < 1/e
        decorrtimes = np.argmax(np.abs(autocorrs[:,numsteps:]) - np.exp(-1) < 0,
                                axis=1)
        
        return autocorrs, decorrtimes
            
        
    
    def heatcapacity(Ts, Es, burns):
        """
        A very simple function that computes the lattice heat capacity 
        according to the fluctuation dissipation theorem. Makes use of the 
        function 'reducedvar' to discard energies pre burn-in.
        """
        varEs = reducedvar(Es, burns, axis=1)
        C = varEs/Ts**2
        return C  
        
    
def threadedfunc(pipe, func):
    """
    Very simple function to change a return command to a send-over-pipe
    command for the argument 'function'.
    """
    out = func()
    if out is not None:
        pipe.send((out))
    
def parallelise(functions):
    """
    Function utilising multiprocessing to execute several 'functions' in 
    parallel. Returns all of the arguments of the functions as a tuple. 
    Spawning pipes and processes can take about a second per process, so
    there is no point parallelising routines that take ~1 sec.

    Parameters
    ----------
    functions : iterable of callables
        List or tuple of functions to execute, must be of the form func() 
        (no arguments), so use functools.partial if necessary.

    Returns
    -------
    returns : tuple
        Tuple of returns of all the 'functions' (which may themselves be 
         tuples).

    """
    n = len(functions)
    pipes = []
    threads = []
    for i in range(n):
        pipes.append(multi.Pipe())
        threads.append(multi.Process(name='func %d'%(i), target=threadedfunc,
                                     args=(pipes[i][1], functions[i])))
        threads[i].start()
    try:
        returns = list(range(n))
        stillgoing = list(range(n))
        while len(stillgoing):
            for i in stillgoing:
                if pipes[i][0].poll(0.1):
                    returns[i] = pipes[i][0].recv()
                    pipes[i][0].close()
                    threads[i].join()
                    stillgoing.remove(i)
            time.sleep(0.01)
        return returns
    except Exception as e:
        for i in range(n):
            threads[i].close()
            pipes[i][0].close()
        raise(e)

def reducedmean(data, numzeros, axis=0):
    """
    A simple function to compute the true mean of the data in an array, 'data'
    that has been padded with a number, 'numzeros', of zeros. Takes an axis
    argument much like np.mean.
    """
    N = data.shape[axis]
    n = numzeros
    redmean = (N/(N - n))*np.mean(data, axis=axis)
    return redmean

def reducedvar(data, numzeros, axis=0):
    """
    A simple function to compute the true varaince of the data in an array,
    'data' that has been padded with a number, 'numzeros', of zeros.
    Takes an axis argument much like np.var.
    """
    N = data.shape[axis]
    n = numzeros
    redvar = (N/(N - n))*(np.var(data, axis=axis) 
                          - (n/(N - n))*np.mean(data, axis=axis)**2)
    return redvar

def rollavg(data, window=6, retain_size=True):
    """
    A simple function to create the rolling average of some input 'data', for
    smoothing plots. This routine creates an avrage at each point from 
    preceding and succeding points, so there is no 'lag time' and the location
    of peaks (but not their height), for example, is reproduced faithfully.

    Parameters
    ----------
    data : numpy array
        1D or 2D numpy array of data to be smoothed. If the array is 2D, the
        axis along which to smooth is taken to be the second axis. i.e an
        M x N array is treated like M independent vectors.
    window : int, optional
        The number of points from which to make each average. This must be an
        even number. The default is 6.
    retain_size : bool, optional
        If retain_size is True, the head and tail of the 'data' are left
        unchanged in order to return an array with the same dimensions as
        'data'. Otherwise, the head and tail are discarded and, if data has 
        dimensions M x N, the output will have dimensions M x (N - window).
        The default is True.

    Returns
    -------
    ret : numpy array
        Smoothed data. The size of this array is determined by 'retain_size'.

    """
    trdata = data.T
    cumul = np.cumsum(trdata, axis=0)
    ravg = (cumul[window:] - cumul[:-window])/window
    if retain_size:
        try:
            ret = np.vstack((trdata[:window//2], ravg,
                             trdata[-window//2:])).T
        except ValueError:
            ret = np.concatenate((trdata[:window//2], ravg,
                                  trdata[-window//2:]))
    else:
        ret = ravg.T
    return ret
        
if __name__ == '__main__':
    ## Doesn't run if this .py is imported as a library.
    
    ## Modify as necessary to make .npz files.
    
    plt.close('all')
    plt.ioff()

    Ns = np.array([32])
    numchains = 50 
    numsteps = 300
    H = 0.0
    Ts = np.linspace(0.1, 2.0, numchains)
    
    motherseed = time.time_ns()
    print('motherseed = ', motherseed)
    seeds = np.array(np.random.SeedSequence(
        motherseed).spawn(numchains*Ns.size)).reshape((Ns.size, numchains))
             
    mags = np.zeros((Ns.size, numchains))
    caps = mags.copy()
    decorrtimes = np.zeros((Ns.size, numchains), dtype=int)
    burns = decorrtimes.copy()
    for i in range(Ns.size):
        print(Ns[i])
        latts = np.array([lattice(Ns[i], H, Ts[m], uniform=True, seed=seeds[i,m])
                          for m in range(numchains)])
        sbars, Es, burns[i,:] = lattice.stepforward_l(latts, numsteps)
        maxburnloc = np.argmax(burns[i,:])
        print('maxburn = %d at T = %.3f'%(burns[i, maxburnloc], 
                                          Ts[maxburnloc]))
        
        for m in range(numchains):
            sbars[m,:burns[i,m]] = 0.0
            Es[m,:burns[i,m]] = 0.0
        mags[i,:] = Ns[i]**2*reducedmean(sbars, burns[i,:], axis=1)
        caps[i,:] = lattice.heatcapacity(Ts[:], Es, burns[i,:])
        autocorrs, decorrtimes[i,:] = lattice.autocorrelation(sbars,
                                                              burns[i,:])
        
    # np.savez('range4N_rangeT_30000steps_uni.npz',
    #           sidelengths=Ns, temps=Ts,
    #           fields=0.0, burntimes=burns, #energies=Es,
    #           magnetisations=mags,
    #           heatcapacities=caps,
    #           #decorrtimes=decorrtimes,
    #           seed=motherseed)


    fig, ax = plt.subplots()
    
    ax.plot(Ts, mags.T, '.', markersize=2)
    
    plt.show()
    

    
    