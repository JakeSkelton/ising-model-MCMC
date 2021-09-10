# -*- coding: utf-8 -*-
"""
comp_project_2_js2443.py

Secondary program of the computing project. Plots how important quantities
evolve in time for several lattices.
"""

import numpy as np
from comp_project_0_js2443 import lattice
import matplotlib.pyplot as plt
import time
import npzviewer
import plotaesthetics


def makeplots(Tlist, var1, var2, burns, spread=False, legend=True):
    """
    Plot the variation of up to two quantities with time for several lattices.

    Parameters
    ----------
    Tlist : numpy array
        Temperatures for each of the M lattices to be plotted.
    var1 : numpy array
        MxN array of data, for M lattices and N timesteps.
    var2 : numpy array
        As with var1.
    burns : numpy array
        Burn-in times for each of the M lattices.
    spread : bool, optional
        If True, the plot traces are displaced vertically from one another, to
        aid visualisation. The default is False.
    legend : bool, optional
        Whether or not to include a key. The default is True.

    Returns
    -------
    figs : list of matplotlib figure objects
        The plotted figures.

    """
    numchains = Tlist.size
    numsteps = var1[0].size
    spread = int(spread)
    
    plt.ioff()
    plt.close('all')

    
    figs = [plt.figure(figsize=(4, 4), dpi=300, tight_layout=True),
            plt.figure(figsize=(4,4))]
    axs = [figs[i].add_subplot(
        111, xlim=[0, numsteps],
        xlabel=('Time step, $t$'),
        ylabel=([r'Mean spin, $\bar{s}(t)$', r'Mean energy, $\bar{E}$',
                 r'$\Delta M_N(t)^2$'][i]),
        yticklabels=[])
        for i in range(2)]
    
    axs[0].set_ylim([-2*spread*(numchains - 1) - 1.1, 1.1])
    #axs[1].set_ylim([-spread*(numchains - 1) - 0.01, 1.01])
    axs[0].minorticks_on()
    
    for m in range(numchains):
        axs[0].plot(var1[m,:] - 2*spread*m, 'k-', lw=0.5)
        axs[0].plot(burns[m], var1[m,burns[m]] - 2*spread*m, 'bx', 
                    markersize=8, label='_nolegend_')
        axs[1].plot(var2[m,:] - spread*m, 'k', lw=0.5)
    
    if legend and numchains <= 20:
        axs[0].legend([r'T=%.2f, burn=%d'
                        %(Tlist[m], burns[m])
                        for m in range(numchains)])
    
    plt.show()    
    return figs
    
if __name__=='__main__':
    
    # filename = '32N_rangeT_10000steps.npz'
    # npzviewer.output(filename)
    # npzobj = np.load(filename)
    
    # inds=slice(10,30,1)
    
    # Ns = npzobj['sidelengths'][inds]
    # Hs = npzobj['Hs'][inds]
    # Ts = npzobj['temps'][inds]
    # burns = npzobj['burns'][inds]
    # sbars = npzobj['sbars'][inds]
    # svars = npzobj['svars'][inds]
    # Es = npzobj['Es'][inds]
    
    motherseed = time.time_ns()
    print('motherseed = ', motherseed)
    
    N = 32
    #numchains = 10
    numsteps = 5000
    H = 0.0
    Ts = np.arange(0.2, 2.0, 0.2)
    numchains = Ts.size
    seeds = np.random.SeedSequence(motherseed).spawn(numchains)
    
    latlist = np.array([lattice(N, H, Ts[m], uniform=False, seed=seeds[m])
                        for m in range(numchains)])
    sbars, Es, burns = lattice.stepforward_l(latlist, numsteps)
    
    out = makeplots(Ts, sbars, Es/N**2, burns,
                    timeseries=True, spread=True, legend=False)
    
    # out[0].savefig('burnin_crosses', dpi=300, bbox_inches='tight',
    #     metadata={'comment': """motherseed = %d, %d chains , 
    #               temps = range(0.2, 2.0, 0.2)"""})
    
    