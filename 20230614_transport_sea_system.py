#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:34:25 2023

@author: hugo
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:27:23 2023

@author: hugo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import pi
from scipy import integrate

from module_nsol_v3 import fd_sol_transport_sea


Q = 1000
A = 7500
k = 1000
kappa = 200
L = 30000
R = 10000
D = 20
N_x = 25
N_r = 25
c7 = 1/4

a = A/(np.pi*D)

m = 101
t = np.linspace(0,1e6,m)

nr, dt = 10000, 1000
n_steps = int(t[-1]/dt)

dr = (R-a)/(nr+1)

r = np.linspace(a,R,nr+2)[1:-1]
axs = r


def ssr0(r):
    pwr = c7*Q/(kappa*pi*D)
    d1 = (0.5-1)/(a**pwr - R**pwr)
    return d1*(r**pwr - R**pwr) + 1
    
def ssr(r):
    pwr = Q/(kappa*pi*D)
    d1 = (0.5-1)/(a**pwr - R**pwr)
    return d1*(r**pwr - R**pwr) + 1


res, T_A, T_D = fd_sol_transport_sea(Q, A, kappa, R, D, nr, n_steps, dt, c7)
t_num = np.arange(0,(n_steps+1)*dt, dt)

s0 = ssr0(r)
sn = ssr(r)

n_skips = int((t[1]-t[0])/dt)


T_A = np.array(T_A)
T_D = np.array(T_D)

maxA = np.max(T_A)
maxD = np.max(T_D)
norm = max(maxA, maxD)

normtot = np.max(np.abs(T_D-T_A))

fig, axes = plt.subplots(1,3, figsize=(16,8))
ax = axes[0]
im = ax.imshow(T_A, extent=[axs[0],axs[-1], t[-1],t[0]], aspect='auto', vmin=0, vmax=norm, label='T_A')
plt.colorbar(im, ax=ax)
ax.set_title('T_A')
ax.set_xlabel('r')
ax.set_ylabel('t')

ax = axes[1]
im = ax.imshow(T_D, extent=[axs[0],axs[-1], t[-1],t[0]], aspect='auto', vmin=0, vmax=norm, label='T_D')
plt.colorbar(im, ax=ax)
ax.set_title('T_D')
ax.set_xlabel('r')
ax.set_ylabel('t')

ax = axes[2]
im = ax.imshow((T_D-T_A), extent=[axs[0],axs[-1], t[-1],t[0]], cmap='bwr', vmin=-normtot, vmax=normtot, aspect='auto', label='T_D-T_A')
plt.colorbar(im, ax=ax)
ax.set_title('T_D-T_A')
ax.set_xlabel('r')
ax.set_ylabel('t')

plt.show()


T = np.arange(0,(n_steps)*dt, dt)
##########################
####### ANMINATION #######
##########################

n_skips = int((t[1]-t[0])/dt)
t_num = np.arange(0,(n_steps)*dt, dt)

fig, axes = plt.subplots(2,1, figsize=(12,12))
ax = axes[0]
ax.plot(axs, s0, color='C0')
ax.plot(axs, sn, color='C0')
ax.set_xlim(a, R)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$s$')

ax = axes[1]
ax.set_xlim(a, R)
# ax.set_ylim(0, ylimit)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$T_A$')

# Plot the surface.
def init():
    ax = axes[0]
    ax.plot(axs, s0, color='C0')
    ax.plot(axs, sn, color='C0')
    ax.set_xlim(a, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$s$')
    
    ax = axes[1]
    ax.set_xlim(a, R)
    # ax.set_ylim(0, ylimit)
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$T_A$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax = axes[0]
    ax.clear()
    ax.plot(axs, s0, color='C0')
    ax.plot(axs, sn, color='C0')
    ax.plot(axs, res[i*n_skips], color='C1')
    ax.set_xlim(a, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$s$')
    ax.set_title('t = ' + str(t[i]) + ' s')
    
    ax = axes[1]
    ax.clear()
    ax.plot(axs, T_A[i*n_skips], color='C0', label='T_A')
    ax.plot(axs, T_D[i*n_skips], color='C1', label='T_D')
    ax.set_xlim(a, R)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$T$')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=m, interval=10, repeat=False)
plt.show()
