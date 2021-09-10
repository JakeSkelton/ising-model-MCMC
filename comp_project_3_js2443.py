# -*- coding: utf-8 -*-
"""
comp_project_3_js2443.py

Secondary program for the computing project. Plots how important quantities
evolve with heat bath temperature and/or lattice size.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import npzviewer
from scipy.optimize import curve_fit
import comp_project_0_js2443 as c0
import plotaesthetics 
from plotaesthetics import plotcols as cols

plt.close('all')
plt.ioff()

def burns(file_uni='range4N_rangeT_30000steps_uni2.npz',
          file_ran='range4N_rangeT_30000steps_ran2.npz'):

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True,
                            figsize=(4,5), dpi=300, tight_layout=True)
    
    for i in range(2):
        file = [file_uni, file_ran][i]
        npzviewer.output(file)
        npzobj = np.load(file)
    
        Ns = npzobj['sidelengths']
        Ts = npzobj['temps']
        numchains = Ts.size
        #decotimes = npzobj['decorrtimes']
        burns = npzobj['burntimes']
        
        axs[i].set(xlim=([0, Ts[-1]]), yscale='log',
                   ylabel='Burn-in time, $t_b$')
        
        axs[i].plot(Ts, burns.T, '.', markersize=2)
        
    axs[1].set_xlabel('Temperature, $T$')
    axs[0].legend(['%d'%N for N in Ns], title='N', fontsize='small')

    plt.show()
    return fig

def autocorrs(filename='auto_range4N_rangeT_30000steps_uni.npz'):
    """
    Function to plot decorrelation and burn-in times vs temperature.
    """
    
    npzviewer.output(filename)
    npzobj = np.load(filename)
    
    Ns = npzobj['sidelengths']
    Ts = npzobj['temps']
    numchains = Ts.size
    decotimes = npzobj['decorrtimes']
    burns = npzobj['burntimes']
    
    ylabels = [r'Decorrelation time, $\tau_e$', 'Burn-in time, $t_b$']
    figs = [plt.figure(figsize=(4,3), dpi=300, tight_layout=True)
                  for i in range(2)]
    ax1, ax2 = [figs[i].add_subplot(
        111, xlim=[0, Ts[-1]], yscale='log',
        xlabel='Temperature, $T$', ylabel=ylabels[i]) for i in range(2)]
    
    ax1.plot(Ts, decotimes.T, '.', markersize=2)
    ax2.plot(Ts, burns.T, '.', markersize=2)
    
    ax1.legend(['%d'%N for N in Ns], title='N')
    ax2.legend(['%d'%N for N in Ns], title='N')

    plt.show()
    return figs
    

def multidata(filename = 'range4N_rangeT_10000steps_uni2.npz'):
    """
    Function to plot variation of magnetisation and heat capacity with 
    temperature, for several lattices.
    """

    npzviewer.output(filename)
    npzobj = np.load(filename)
    
    Ns = npzobj['sidelengths'] ## may need to turn into array if N is int
    #Hs = npzobj['Hs']
    Ts = npzobj['temps']
    numchains = Ts.size
    mags = (npzobj['magnetisations'].T/Ns**2).T
    caps = (npzobj['heatcapacities'].T/Ns**2).T
    #decotimes = npzobj['decorrtimes']
    burns = npzobj['burntimes']
    
    wind = 4
    avgs_m = c0.rollavg(mags, window=wind)
    avgs_c = c0.rollavg(caps, window=wind)
    
    fig1, axs1 = plt.subplots(2, sharex=True, figsize=(4, 5), dpi=300,
                              tight_layout=True)
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ylabels = [r'Magnetisation $\langle M \rangle /N^2$',
               r'Heat capacity, $C/N^2$',
               r'Burn time, $t_b$',
               r'Decorrelation time, $\tau_e$']
    xlabels=['','Temperature, $T$']
    for i in range(2):
        axs1[i].set(xlim=[0, Ts[-1]], xlabel=xlabels[i], ylabel=ylabels[i])
    ax2.set(xlim=[0, Ts[-1]], xlabel=xlabels[1], ylabel=ylabels[2])

    msize, lw = 2, 1
    for i in range(Ns.size):
        axs1[0].plot(Ts, mags[i].T, '+', c=cols[i], markersize=msize)
        #axs1[0].plot(Ts, avgs_m[i].T, '-', c=cols[i],lw=lw)
        axs1[1].plot(Ts, caps[i].T, '+', c=cols[i], markersize=msize,
                     label='_nolegend_')
        axs1[1].plot(Ts, avgs_c[i].T, '-', c=cols[i], lw=lw)
        ax2.plot(Ts, burns[i].T, '.', markersize=msize)
    
    axs1[1].legend(['%d'%N for N in Ns], title='N', fontsize='small')
    # axs1[0].legend(['%.2f'%H for H in Hs], title='H', fontsize='small')
    # ax2.legend(['%.2f'%H for H in Hs], title='H')
    
    plt.show()
    return fig1, fig2
    
def capsonly(filename='Tcfocus_rangeN_5000steps.npz'):
    """
    Plot heat capacity vs temperature, as well as rolling average.
    """
    
    npzviewer.output(filename)
    npzobj = np.load(filename)
    
    Ns = np.array([npzobj['sidelength']])
    Hs = npzobj['fields']
    Ts = npzobj['temps']
    numchains = Ts.size
    caps = npzobj['heatcapacities']
    
    smoothed = c0.rollavg(caps, window=6)
    Tcs = Ts[np.argmax(smoothed, axis=1)]
    
    fig = plt.figure(figsize=(16,9))
    axs = []
    for i in range(2):
        axs.append(fig.add_subplot(121+i, xlim=[Ts[0], Ts[-1]],
                                   xlabel='Temperature, $T$',
                                   ylabel=r'Heat capacity, $C/N^2$'))
    
    axs[0].plot(Ts.T, caps.T/Ns**2, '.', markersize=2)
    axs[1].plot(Ts.T, smoothed.T/Ns**2, '-')

    #axs[0].legend(['N = %d'%N for N in Ns])
    axs[0].legend(['H = %.2f'%H for H in Hs])
    axs[1].legend(['$T_c$ = %1.3f'%Tc for Tc in Tcs])
    
    plt.show()
    return fig

def argmaxwitherr(data):
    """
    Simple function to return usual np.argmax, as well as the second highest
    to the left and right.
    """
    argmax = np.argmax(data, axis=1)
    arg2max_l = np.zeros(data.shape[0], dtype=int)
    arg2max_r = arg2max_l.copy()
    for i in range(data.shape[0]):
        arg2max_r[i] = argmax[i] + np.argmax(data[i,argmax[i]+1:])
        arg2max_l[i] = np.argmax(data[i,:argmax[i]])
    
    return(argmax, [arg2max_l, arg2max_r])


def scaling(filename=''):
    """
    Plot the variation of the temperature at which heat capacity is maximised
    vs lattice size, fit the canonical power law scaling relationship, and 
    thus estimate the critical temperature of an infinite lattice.
    """
    npzviewer.output(filename)
    npzobj = np.load(filename)
    
    Ns = npzobj['sidelengths']
    #Hs = npzobj['Hs']
    Ts = npzobj['temps']
    numchains = Ts.shape[0]
    caps = (npzobj['heatcapacities'].T/Ns**2).T
    
    wind = 20
    smoothed = c0.rollavg(caps, window=wind, retain_size=False)
    argmax, interval = argmaxwitherr(caps)
    print(interval)
    Tcs = Ts[np.arange(Ns.size), argmax]
    Tcs_err = np.array([Ts[np.arange(Ns.size), bound] for bound in interval])
    
    print('Tcs = ', Tcs)
    
    fitfunc = lambda x, a, b, c: c + a*x**(-1/b) 
    fit = curve_fit(fitfunc, Ns, Tcs, sigma=np.abs(Tcs_err - Tcs).max(axis=0))
    params = fit[0]
    errs = np.sqrt(np.diag(fit[1]))
    print(r'T_c = %1.4f +/- %1.4f i.e. [%1.4f, %1.4f]'
          %(params[2], errs[2], params[2] - errs[2], params[2] + errs[2]))
    
    figs = [plt.figure(figsize=(4,4), dpi=300, tight_layout=True)
            for i in range(2)]
    axs = []
    xlabels = ['Temperature, $T$', 'Lattice width, $N$']
    ylabels = [r'Heat capacity, $C/N^2$', 'Critical temperature, $T_c$']
    for i in range(2):
        axs.append(figs[i].add_subplot(111, xlabel=xlabels[i],
                                       ylabel=ylabels[i]))
    
    axs[1].set_xscale('log')
    axs[0].plot(Ts.T, caps.T, '.', markersize=2)
    axs[1].plot(Ns, Tcs, 'kx', label='_nolegend_')
    axs[1].errorbar(Ns, Tcs, np.abs(Tcs_err - Tcs), ls='', c='0.6',
                    label='_nolegend_')
    ns = np.logspace(2, 7, base=2)
    axs[1].plot(ns, fitfunc(ns, fit[0][0], fit[0][1], fit[0][2]), 'k-')
    
    axs[1].legend(['Fit of $y = c + ax^{-1/b}$ \n' + 
                   'a = %.4f $\pm$ %.4f \n'%(params[0], errs[0]) +
                   'b = %.4f $\pm$ %.4f \n'%(params[1], errs[1]) +
                   'c = %.4f $\pm$ %.4f'%(params[2], errs[2])])
    
    plt.show()
    return figs

    
def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
    

def hysteresis(filename='hyst_32N_rangeT_rangeH_10000steps.npz', ind=(0,0)):
    """
    Make a multi-panel figure of M vs H hysteresis plots, and energy vs H plots,
    with a colour map.
    """
    npzviewer.output(filename)
    npzobj = np.load(filename)
    
    N = npzobj['sidelength']
    numchains = npzobj['numchains']
    Hs = npzobj['Htrajectories']
    Ts = npzobj['temps']
    mags = npzobj['magnetisations']
    Es = npzobj['energies']
    cycle = 3.0
    
    print('T = ', Ts[ind[1]])
    
    numsteps = Hs.shape[1]
    numTs = np.unique(Ts).size
    Hs = np.array(np.split(Hs, numTs, axis=0))[ind]
    mags = np.array(np.split(mags, numTs, axis=0))[ind]
    Es = np.array(np.split(Es, numTs, axis=0))[ind]
    Hmax = np.max(Hs)
    
    fig,axs = plt.subplots(2, 3, sharex='col', sharey='row',
                           figsize=(8,5), dpi=300, tight_layout=True)
    ylabels = [r'Mean spin, $\bar{s}$',
                r'Energy per site, $\bar{E}$']
    titles = [r'$T$ = %1.2f'%(T) for T in Ts[ind[1]]]
    for i in range(3):
        for j in range(2):
            axs[j,i].set(xlim=[-Hmax-0.05, Hmax+0.05])
            axs[j,0].set(ylabel=ylabels[j])
        axs[0,i].set(ylim=[-1.1, 1.1], title=titles[i])
        axs[1,i].set_xlabel('Applied field, $H$')
    
        norm=plt.Normalize(vmin=0, vmax=cycle)
        data = make_segments(Hs[i].T, mags[i].T/N**2)
        lc = LineCollection(data, cmap='winter', norm=norm, linewidth=0.8, 
                            array=np.linspace(0, cycle, numsteps))
        
        axs[0,i].add_collection(lc)   
        #axs[0].plot(Hs.T, mags.T/N**2, 'k', lw=0.5)
        axs[1,i].scatter(Hs[i].T, Es[i].T/N**2, s=0.1, marker='+',
                         cmap='winter', c=np.linspace(0, cycle, numsteps))

    plt.show()
    
    return fig
    
if __name__=='__main__':  
    
    #fig = burns()
    
    #figs = autocorrs()

    #capsonly('rangeH_32N_rangeT_10000steps.npz')
    
    #figs = scaling('Tcfocus_rangeN_30000steps2.npz')
    
    figs = multidata('range5N_rangeT_30000steps_uni.npz')
    
    #fig = hysteresis(ind=(2, slice(2,7,2)))
    
    # figs[0].savefig(
    #     'caps_mags_rangeN2.png', dpi=300,
    #     bbox_inches='tight', 
    #     metadata={'comment': """30000 steps. 5 different Ns. Made with file 
    #               range5N_rangeT_30000steps_uni.npz"""})
    
        