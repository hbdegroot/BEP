#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:10:44 2023

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
from module_sol_v1 import eigenvalues_x
from module_sol_v1 import eigenvalues_r

default = '1'
Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r = def_func(default)

time_scale = tc(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)
print('time_scale =', time_scale)
print('expected e-folding =', 1/time_scale)

a = A/(np.pi*D)

m = 1000

f = 1/1000

x = np.linspace(0,L,100)
r = np.linspace(a,R,100)
t = np.linspace(0,1e6,m)


result1, sx_arr1, sr_arr1, transx1, transr1 =  sol(x, r, t, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, gamma=time_scale)
result2, sx_arr2, sr_arr2, transx2, transr2 =  sol(x, r, t, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, gamma=time_scale*f)

############## Analysis
axs = np.append(-x[::-1], r)
s0 = np.append(ssx0(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))
sn = np.append(ssx(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))

deltas = sn - s0

deltat1 = np.zeros(len(axs))
deltat2 = np.zeros(len(axs))
for i in range(len(axs)):
    for j in range(len(t)):
        if sn[i] - result1[j][i] <= deltas[i]/np.e and deltat1[i] == 0:
            deltat1[i] = t[j]
        if sn[i] - result2[j][i] <= deltas[i]/np.e and deltat2[i] == 0:
            deltat2[i] = t[j]
    

Tx = 1/eigenvalues_x(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)
Tr = 1/eigenvalues_r(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)
# Tx = 1/(eigenvalues_x(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)-time_scale)
# Tr = 1/(eigenvalues_r(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)-time_scale)

############## PLOTTING

fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(axs, deltat1, 'C0-', label='1x')
ax.plot(axs, deltat2, 'C1-', label=str(f) + 'x')
ax.hlines(Tx, -L,0, colors='k')
ax.hlines(Tr, a,R, colors='c')
ax.hlines(1/time_scale, -L, R, colors='C0')
ax.hlines(1/(f*time_scale), -L, R, colors='C1')
ax.set_xlim(-L, R)
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'e-folding $t$')
ax.legend()
plt.show()

# #-----------------------------------------------

# fig, axes = plt.subplots(2,1, figsize=(8,8))

# ax = axes[0]
# ax.plot(axs, s0)
# ax.plot(axs, sn)
# ax.set_xlim(-L, R)
# ax.set_ylim(0, 1)
# ax.set_xlabel(r'$-x & r$')
# ax.set_ylabel(r'$s$')

# ax = axes[1]
# ax.set_xlim(0, t[-1])
# # ax.set_ylim(0, ylimit)
# ax.set_xlabel(r'$t$')
# ax.set_ylabel(r'$ds/dx & ds/dr$')

# # Plot the surface.
# def init():
#     ax = axes[0]
#     ax.plot(axs, s0, label='initial')
#     ax.plot(axs, sn, label='final')
#     ax.set_xlim(-L, R)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel(r'$x$')
#     ax.set_ylabel(r'$s$')
    
#     ax = axes[1]
#     ax.set_xlim(0, t[-1])
#     # ax.set_ylim(0, ylimit)
#     ax.set_xlabel(r'$t$')
#     ax.set_ylabel(r'$ds/dx & ds/dr$')
#     return 

# # animation function.  This is called sequentially
# def animate(i):
#     ax = axes[0]
#     ax.clear()
#     ax.plot(axs, s0, label='initial')
#     ax.plot(axs, sn, label='final')
#     ax.plot(axs, result1[i], 'r-', label=str(int(t[i])))
#     ax.plot(axs, result2[i], 'k-', label=str(int(t[i])))
#     ax.plot(axs[len(x)], result1[i][len(x)], 'r.')
#     ax.plot(axs[len(x)], result2[i][len(x)], 'k.')
#     ax.plot(0, f(t[i]), 'ro')
#     ax.set_xlim(-L, R)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel(r'$x$')
#     ax.set_ylabel(r'$s$')
#     ax.legend()
    
#     ax = axes[1]
#     ax.clear()
#     # ax.plot(t[:i], sx_arr[:i], label='s_x')
#     # ax.plot(t[:i], sr_arr[:i], label='s_r')

#     # ax.plot(t[:i], -transx[:i], 'r--', label='transx')
#     # ax.plot(t[:i], transr[:i], 'k--', label='transr')
#     ax.set_xlim(0, t[-1])
#     # ax.set_ylim(0, ylimit)
#     ax.set_xlabel(r'$t$')
#     ax.set_ylabel(r'$ds/dx & ds/dr$')
#     ax.legend()
#     return 

# anim = FuncAnimation(fig, animate, frames=m, interval=10, repeat=False)
# plt.show()