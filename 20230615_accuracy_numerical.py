#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:13:41 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import pi

from module_nsol_v3 import fd_sol_faster
from module_nsol_v3 import fd_sol_horde


Q = 500
A = 7500
k = 900
kappa = 180
L = 45000
R = 7000
D = 20
N_x = 25
N_r = 25
c7 = 0.5

a = A/(np.pi*D)

m = 101
t = np.linspace(0,1e6,m)

nx, nr, dt = 500, 500, 1000
n_steps = int(t[-1]/dt)

dt2 = dt/2
n_steps2 = int(t[-1]/dt2)

dt3 = dt/10
n_steps3 = int(t[-1]/dt3)

dx = L/(nx+1)
dr = (R-a)/(nr+1)

x = np.linspace(0,L,nx+2)[1:-1]
r = np.linspace(a,R,nr+2)[1:-1]
axs = np.append(-x[::-1], r)

nx2, nr2 = 2*nx, 2*nr
dx2 = L/(nx2+1)
dr2 = (R-a)/(nr2+1)
x2 = np.linspace(0,L,nx2+2)[1:-1]
r2 = np.linspace(a,R,nr2+2)[1:-1]
axs2 = np.append(-x2[::-1], r2)

nx3, nr3 = 10*nx, 10*nr
dx3 = L/(nx3+1)
dr3 = (R-a)/(nr3+1)
x3 = np.linspace(0,L,nx3+2)[1:-1]
r3 = np.linspace(a,R,nr3+2)[1:-1]
axs3 = np.append(-x3[::-1], r3)


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


res11 = fd_sol_faster(Q, A, k, kappa, R, L, D, nx, nr, n_steps, dt, c7)
res21 = fd_sol_faster(Q, A, k, kappa, R, L, D, nx, nx, n_steps*2, dt/2, c7)
res31 = fd_sol_faster(Q, A, k, kappa, R, L, D, nx, nr, n_steps*10, dt/10, c7)

res12 = fd_sol_faster(Q, A, k, kappa, R, L, D, nx2, nr2, n_steps, dt, c7)
res22 = fd_sol_faster(Q, A, k, kappa, R, L, D, nx2, nx2, n_steps*2, dt/2, c7)
res32 = fd_sol_faster(Q, A, k, kappa, R, L, D, nx2, nr2, n_steps*10, dt/10, c7)

res13 = fd_sol_faster(Q, A, k, kappa, R, L, D, nx3, nr3, n_steps, dt, c7)
res23 = fd_sol_faster(Q, A, k, kappa, R, L, D, nx3, nx3, n_steps*2, dt/2, c7)
res33 = fd_sol_faster(Q, A, k, kappa, R, L, D, nx3, nr3, n_steps*10, dt/10, c7)

sn = np.append(ssx(x, Q, A, k, kappa, R, L, D)[::-1], ssr(r, Q, A, k, kappa, R, L, D))
sn2 = np.append(ssx(x2, Q, A, k, kappa, R, L, D)[::-1], ssr(r2, Q, A, k, kappa, R, L, D))
sn3 = np.append(ssx(x3, Q, A, k, kappa, R, L, D)[::-1], ssr(r3, Q, A, k, kappa, R, L, D))

fig, axes = plt.subplots(3,1, figsize=(16,8))
ax = axes[0]
ax.plot(axs, sn-res11, color='C1', ls='-', label=r'$\Delta t = $' + str(int(dt)) + 's')
ax.plot(axs, sn-res21, color='C2', ls='--', label=r'$\Delta t = $' + str(int(dt/2)) + 's')
ax.plot(axs, sn-res31, color='C3', ls=':', label=r'$\Delta t = $' + str(int(dt/10)) + 's')
ax.set_xlabel(r'$-x&r$')
ax.set_ylabel(r'$\Delta \zeta$')
ax.set_title(r'$\Delta x = $' + str(dx) + 'm, $\Delta r =$ ' + str(dr))
ax.legend()

ax = axes[1]
ax.plot(axs2, sn2-res12, color='C1', ls='-', label=r'$\Delta t = $' + str(int(dt)) + 's')
ax.plot(axs2, sn2-res22, color='C2', ls='--', label=r'$\Delta t = $' + str(int(dt/2)) + 's')
ax.plot(axs2, sn2-res32, color='C3', ls=':', label=r'$\Delta t = $' + str(int(dt/10)) + 's')
ax.set_xlabel(r'$-x&r$')
ax.set_ylabel(r'$\Delta \zeta$')
ax.set_title(r'$\Delta x = $' + str(dx2) + 'm, $\Delta r =$ ' + str(dr2))
ax.legend()


ax = axes[2]
ax.plot(axs3, sn3-res13, color='C1', ls='-', label=r'$\Delta t = $' + str(int(dt)) + 's')
ax.plot(axs3, sn3-res23, color='C2', ls='--', label=r'$\Delta t = $' + str(int(dt/2)) + 's')
ax.plot(axs3, sn3-res33, color='C3', ls=':', label=r'$\Delta t = $' + str(int(dt/10)) + 's')
ax.set_xlabel(r'$-x&r$')
ax.set_ylabel(r'$\Delta \zeta$')
ax.set_title(r'$\Delta x = $' + str(dx3) + 'm, $\Delta r =$ ' + str(dr3))
ax.legend()
plt.show()

import matplotlib.patches as mpatches
import matplotlib.lines as mlines


fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(axs, sn-res11, color='C0', ls='-')
ax.plot(axs, sn-res21, color='C0', ls='--')
ax.plot(axs, sn-res31, color='C0', ls=':')

ax.plot(axs2, sn2-res12, color='C1', ls='-')
ax.plot(axs2, sn2-res22, color='C1', ls='--')
ax.plot(axs2, sn2-res32, color='C1', ls=':')

ax.plot(axs3, sn3-res13, color='C2', ls='-')
ax.plot(axs3, sn3-res23, color='C2', ls='--')
ax.plot(axs3, sn3-res33, color='C2', ls=':')

ax.set_xlabel(r'$-x&r$ (m)')
ax.set_ylabel(r'$[s(\infty) - s(10^6$s$)]/s_{sea}$')
#ax.set_title(r'$\Delta x = $' + str(dx3) + 'm, $\Delta r =$ ' + str(dr3))

dt_line = mlines.Line2D([], [], color='k', ls='-', label=r'$\Delta t = $' + str(int(dt)) + 's')
dt2_line = mlines.Line2D([], [], color='k', ls='--', label=r'$\Delta t = $' + str(int(dt/2)) + 's')
dt3_line = mlines.Line2D([], [], color='k', ls=':', label=r'$\Delta t = $' + str(int(dt/10)) + 's')

c0_patch = mpatches.Patch(color='C0', label=r'$\Delta x= $' + str(round(dx)) + 'm, $\Delta r =$ ' + str(round(dr,1)) + 'm')
c1_patch = mpatches.Patch(color='C1', label=r'$\Delta x= $' + str(round(dx2)) + 'm, $\Delta r =$ ' + str(round(dr2,1)) + 'm')
c2_patch = mpatches.Patch(color='C2', label=r'$\Delta x= $' + str(round(dx3)) + 'm, $\Delta r =$ ' + str(round(dr3,1)) + 'm')
plt.legend(handles=[dt_line, dt2_line, dt3_line, c0_patch, c1_patch, c2_patch])

#ax.legend()
plt.show()
