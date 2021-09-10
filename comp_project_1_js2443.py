# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:40:45 2021

comp_project_1_js2443.py

Secondary program for the computing project. Creates an animation of the square
lattice's spin dynamics over time.
"""
import numpy as np
import matplotlib.pyplot as plt
from comp_project_0_js2443 import lattice
from matplotlib.animation import FuncAnimation
import plotaesthetics
from functools import partial
import time


def init_anim(latts, images, texts):
    """
    Artist initialisation for the matplotlib FuncAnimation method.
    """
    global n
    for i in range(n):
        images[i].set_data(latts[i].spins[::-1,:])
        texts[i].set_text(
            r'$t$=%4d, $\bar{s}$=%+1.3f, $\Delta s^2$=%1.3f, $\bar{E}$=%+1.3f'
            %(latts[i].time, np.mean(latts[i].spins),
              np.var(latts[i].spins), latts[i].updateE()/latts[i].N**2))
    return (*texts, *images) 

def update(latts, images, texts, jump, frame_no):
    """
    Frame updater for the matplotlib FuncAnimation method. Relies on
    the 'stepforward' method of the 'lattice' class.
    """
    global n
    for i in range(n):
        latts[i].stepforward(jump)
        images[i].set_data(latts[i].spins[::-1,:])
        texts[i].set_text(
            r'$t$=%4d, $\bar{s}$=%+1.3f, $\Delta s^2$=%1.3f, $\bar{E}$=%+1.3f'
            %(latts[i].time, np.mean(latts[i].spins),
              np.var(latts[i].spins), latts[i].updateE()/latts[i].N**2))
    return (*texts, *images) 
        
if __name__=='__main__':
    n = 1
    N = 500
    H = 0.0
    T = 1.14
    jump = 10
    numframes = None
    motherseed = time.time_ns()
    print('motherseed = ', motherseed)
    seeds = np.random.SeedSequence(motherseed).spawn(n)
    latts = [lattice(N, H, T, uniform=False, seed=seeds[i]) for i in range(n)]
    
    plt.close('all')
    plt.ioff()
    fig = plt.figure(figsize=(10,10), dpi=100, tight_layout=True)
    axs, ims, txts = [], [], []
    for i in range(n):
        axs.append(fig.add_subplot(
            int(np.sqrt(n)), int(np.sqrt(n)), i+1,
            xlim=([-0.5, N - 0.5]), ylim=([-0.5, N - 0.5]),
            xticks=([-0.5, N - 0.5]), yticks=([-0.5, N - 0.5]),
            xticklabels=(['0',str(N)]), yticklabels=(['0', str(N)]), 
            title=('$T$ = %1.2f, $H$ = %+1.2f'%(T, H))))
    
        ims.append(axs[i].imshow(latts[i].spins[::-1,:], 'gray', vmin=0, vmax=1,
                                 interpolation='none'))
        txts.append(axs[i].text(0.03, 0.95, '', transform=axs[i].transAxes, 
                    bbox=dict(facecolor='w'), fontsize='small'))
    
    frames = range(numframes) if numframes else None
    ani = FuncAnimation(fig, partial(update, latts, ims, txts, jump), 
                        init_func=partial(init_anim, latts, ims, txts),
                        interval=1, repeat=False, blit=True, frames=frames,
                        cache_frame_data=False)
    
    # ani.save('metastable_64N_0.1T.mp4', fps=20, dpi=300, 
    #           progress_callback=lambda a, b: print('%3d out of %3d frames saved'
    #                                               %(a, b)),
    #           metadata={'comment':'N = %d, motherseed = %d, T = %f, H= %f'
    #                     %(N, motherseed, T, H)})
    
    # latts[0].stepforward(6400, 'spins')
    # ims[0].set_data(latts[0].spins[::-1,:])
    # txts[0].set_text(
    #     r'$t$=%4d, $\bar{s}$=%+1.3f, $\Delta s^2$=%1.3f, $\bar{E}$=%+1.3f'
    #     %(latts[0].time, np.mean(latts[0].spins), np.var(latts[0].spins),
    #       latts[0].updateE()/latts[i].N**2))
    # fig.savefig('256N_critT_eq2.png', dpi=300, bbox_inches='tight', 
    #             metadata={'comment': 'motherseed = %d, T = %f'
    #                       %(motherseed, T)})
    plt.show()
    


    