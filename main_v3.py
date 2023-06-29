#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:40:52 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from module_sol_v1 import sol
from module_sol_v1 import ssx0
from module_sol_v1 import ssx
from module_sol_v1 import ssr0
from module_sol_v1 import ssr
from module_sol_v1 import def_func
from module_sol_v1 import eigenvalues_x
from module_sol_v1 import eigenvalues_r

from module_nsol_v2 import fd_sol
from module_nsol_v2 import kranenburg


default = '1'
Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r = def_func(default)

tsx, tsr, ts = kranenburg(Q, A, k, kappa, R, L, D)
gamma = 1/ts

a = A/(np.pi*D)

m = 401

x = np.linspace(0,L,1000)
r = np.linspace(a,R,100)
t = np.linspace(0,1e5,m)

nx, nr, dt = 100, 100, 10
n_steps = int(t[-1]/dt)


result, dsdx, dsdr, ndsdx, ndsdr =  sol(x, r, t, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, gamma)
fdres, fdaxs, pl1, pl2, fpl = fd_sol(Q, A, k, kappa, R, L, D, nx, nr, n_steps, dt)


axs = np.append(-x[::-1], r)
s0 = np.append(ssx0(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))
sn = np.append(ssx(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))


############ Analysis of time_scales
x_fd = np.linspace(0,L,nx+1)[:-1]
r_fd = np.linspace(a,R,nr+1)[:-1]


axs = np.append(-x[::-1], r)
s0 = np.append(ssx0(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))
sn = np.append(ssx(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))
s0_fd = np.append(ssx0(x_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr0(r_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))
sn_fd = np.append(ssx(x_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr(r_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))

deltas = sn - s0
deltas_fd = sn_fd - s0_fd

deltat = np.zeros(len(axs))
deltat_fd = np.zeros(len(fdaxs))

print('Calculating e-folding...')
for i in range(len(axs)):
    for j in range(len(t)):
        if sn[i] - result[j][i] <= deltas[i]/np.e and deltat[i] == 0:
            deltat[i] = t[j]
print('Calculating e-folding...')
for i in range(len(fdaxs)):
    for j in range(0, n_steps, n_steps//m):
        if sn_fd[i] - fdres[j][i] <= deltas_fd[i]/np.e and deltat_fd[i] == 0:
            deltat_fd[i] = j*dt
print('Calculating e-folding...') 
Tx = 1/eigenvalues_x(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)
Tr = 1/eigenvalues_r(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)
# Tx = 1/(eigenvalues_x(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)-time_scale)
# Tr = 1/(eigenvalues_r(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)-time_scale)

fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(axs, deltat, 'C0-', label='eigen expansion')
ax.plot(fdaxs, deltat_fd, 'C1-', label='finite difference')
ax.hlines(Tx, -L,0, colors='k')
ax.hlines(Tr, a,R, colors='c')
ax.hlines(1/gamma, -L, R, colors='C0')
# ax.hlines(tsx, -L, 0, colors='r')
# ax.hlines(tsr, a, R, colors='r')
ax.set_xlim(-L, R)
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'e-folding $t$')
ax.legend()
plt.show()

#np.savetxt('save_dir/data/20230522_res_def1_200_200_002.csv', fdres)
####### ANMINATION

n_skips = int((t[1]-t[0])/dt)

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
    ax.plot(axs, result[i], label = 'eigenfunction expansion')
    ax.plot(fdaxs, fdres[i*n_skips], 'r.', label = 'finite difference')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.set_title('t=' + str(t[i]))
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

anim = FuncAnimation(fig, animate, frames=m-1, interval=10, repeat=False)
plt.show()