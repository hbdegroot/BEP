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

from module_nsol_v3 import fd_sol_transport


Q = 500
A = 7500
k = 900
kappa = 180
L = 45000
R = 7000
D = 20
N_x = 25
N_r = 25
c7 = 2

a = A/(np.pi*D)

m = 101
t = np.linspace(0,1e6,m)

nx, nr, dt = 1000, 9000, 500
n_steps = int(t[-1]/dt)

dx = L/(nx+1)
dr = (R-a)/(nr+1)

x = np.linspace(0,L,nx+2)[1:-1]
r = np.linspace(a,R,nr+2)[1:-1]
axs = np.append(-x[::-1], r)

def ssx0(x, Q, A, k, kappa, R, L, D, c7):
    pwr = c7*Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-c7*Q*L/(k*A)))
    c1 = d1*a**pwr
    c2 = -np.exp(-c7*Q*L/(k*A))*c1
    return c1 * np.exp(-c7*Q*x/(k*A)) + c2

def ssr0(r, Q, A, k, kappa, R, L, D, c7):
    pwr = c7*Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-c7*Q*L/(k*A)))
    d2 = 1-d1*R**pwr
    return d1*r**pwr + d2

def ssx(x, Q, A, k, kappa, R, L, D):
    a = A/(pi*D)
    pwr = Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
    c1 = d1*a**pwr
    c2 = -np.exp(-Q*L/(k*A))*c1
    return c1 * np.exp(-Q*x/(k*A)) + c2
    
def ssr(r, Q, A, k, kappa, R, L, D):
    a = A/(pi*D)
    pwr = Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
    d2 = 1-d1*R**pwr
    return d1*r**pwr + d2


res, T_A, T_D = fd_sol_transport(Q, A, k, kappa, R, L, D, nx, nr, n_steps, dt, c7)
t_num = np.arange(0,(n_steps+1)*dt, dt)

s0 = np.append(ssx0(x, Q, A, k, kappa, R, L, D, c7)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, c7))
sn = np.append(ssx(x, Q, A, k, kappa, R, L, D)[::-1], ssr(r, Q, A, k, kappa, R, L, D))

n_skips = int((t[1]-t[0])/dt)


T_A = np.array(T_A)
T_D = np.array(T_D)

maxA = np.max(T_A)
maxD = np.max(T_D)
norm = max(maxA, maxD)

fig, axes = plt.subplots(1,3, figsize=(16,8))
ax = axes[0]
im = ax.imshow(T_A, extent=[axs[0],axs[-1], t[-1],t[0]], aspect='auto', vmin=0, vmax=norm, label='T_A')
plt.colorbar(im, ax=ax)
ax.set_title('T_A')
ax.set_xlabel('-x&r')
ax.set_ylabel('t')

ax = axes[1]
im = ax.imshow(T_D, extent=[axs[0],axs[-1], t[-1],t[0]], aspect='auto', vmin=0, vmax=norm, label='T_A')
plt.colorbar(im, ax=ax)
ax.set_title('T_D')
ax.set_xlabel('-x&r')
ax.set_ylabel('t')

ax = axes[2]
im = ax.imshow((T_D-T_A), extent=[axs[0],axs[-1], t[-1],t[0]], cmap='binary', aspect='auto', label='T_D-T_A')
plt.colorbar(im, ax=ax)
ax.set_title('T_D-T_A')
ax.set_xlabel('-x&r')
ax.set_ylabel('t')

plt.show()


T =  np.arange(0,(n_steps)*dt, dt)
##########################
####### ANMINATION #######
##########################

n_skips = int((t[1]-t[0])/dt)
t_num = np.arange(0,(n_steps)*dt, dt)

fig, axes = plt.subplots(2,1, figsize=(12,12))
ax = axes[0]
ax.plot(axs, s0, color='C0')
ax.plot(axs, sn, color='C0')
ax.set_xlim(-L, R)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'$s$')

ax = axes[1]
ax.set_xlim(-L, R)
# ax.set_ylim(0, ylimit)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$T_A$')

# Plot the surface.
def init():
    ax = axes[0]
    ax.plot(axs, s0, color='C0')
    ax.plot(axs, sn, color='C0')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$-x & r$')
    ax.set_ylabel(r'$s$')
    
    ax = axes[1]
    ax.set_xlim(-L, R)
    # ax.set_ylim(0, ylimit)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$T_A$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax = axes[0]
    ax.clear()
    ax.plot(axs, s0, color='C0')
    ax.plot(axs, sn, color='C0')
    ax.plot(axs, res[i*n_skips], color='C1')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$-x & r$')
    ax.set_ylabel(r'$s$')
    ax.set_title('t = ' + str(t[i]) + ' s')
    
    ax = axes[1]
    ax.clear()
    ax.plot(axs, T_A[i*n_skips], color='C0', label='T_A')
    ax.plot(axs, T_D[i*n_skips], color='C1', label='T_D')
    ax.set_xlim(-L, R)
    ax.set_xlabel(r'$-x&r$')
    ax.set_ylabel(r'$T$')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=m, interval=10, repeat=False)
plt.show()
