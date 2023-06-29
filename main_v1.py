#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:35:29 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from module_sol_v1 import sol
from module_sol_v1 import tc

from module_sol_v1 import ssx0
from module_sol_v1 import ssx
from module_sol_v1 import ssr0
from module_sol_v1 import ssr
from module_sol_v1 import def_func


default = '3'
Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r = def_func(default)

gamma = 5.6e4 #tc(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)

a = A/(np.pi*D)

m = 100

x = np.linspace(0,L,100)
r = np.linspace(a,R,100)
t = np.linspace(0,1e5,m)


result, dsdx, dsdr, ndsdx, ndsdr =  sol(x, r, t, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, gamma)


############## PLOTTING

axs = np.append(-x[::-1], r)
s0 = np.append(ssx0(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))
sn = np.append(ssx(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))


fig, axes = plt.subplots(2,1, figsize=(8,8))

ax = axes[0]
ax.plot(axs, s0)
ax.plot(axs, sn)
ax.set_xlim(-L, R)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'$s$')

ax = axes[1]
ax.set_xlim(0, t[-1])
# ax.set_ylim(0, ylimit)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$ds/dx & ds/dr$')

# Plot the surface.
def init():
    ax = axes[0]
    ax.plot(axs, s0, label='initial')
    ax.plot(axs, sn, label='final')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    
    ax = axes[1]
    ax.set_xlim(0, t[-1])
    # ax.set_ylim(0, ylimit)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax = axes[0]
    ax.clear()
    ax.plot(axs, s0, label='initial')
    ax.plot(axs, sn, label='final')
    ax.plot(axs, result[i], label=str(t[i]))
    ax.plot(axs[len(x)], result[i][len(x)], 'r.')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.legend()
    
    ax = axes[1]
    ax.clear()
    ax.plot(t[:i], -A*k*dsdx[:i], 'r-', label=r'$A k s_x(0)$')
    ax.plot(t[:i], A*kappa*dsdr[:i],'k-', label=r'$A \kappa s_r(a)$')

    ax.plot(t[:i], A*k*ndsdx[:i], 'r--', label='num')
    ax.plot(t[:i], A*kappa*ndsdr[:i], 'k--', label='num')
    
    ax.hlines(Q*result[i][len(x)-1],0,t[-1], label=r'$Q s(0)$')
    
    ax.set_xlim(0, t[-1])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=m, interval=10, repeat=False)
plt.show()