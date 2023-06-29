#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:51:20 2023

@author: hugo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from module_sol_v1 import sol
from module_sol_v1 import sol_numf
from module_sol_v1 import ssx0
from module_sol_v1 import ssx
from module_sol_v1 import ssr0
from module_sol_v1 import ssr
from module_sol_v1 import def_func
from module_sol_v1 import eigenvalues_x
from module_sol_v1 import eigenvalues_r
from module_sol_v1 import goodrootfinder

from module_nsol_v2 import fd_sol
from module_nsol_v2 import kranenburg


default = '1'
Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r = def_func(default)

Q = 1000
A = 2000
k = 8000
kappa = 800
L = 40000
R = 9000
D = 10
N_x = 25
N_r = 25

l0, lmax, fine = 1e-5, 0.1, 400
good_roots_r = goodrootfinder(N_r, l0, lmax, fine, Q, A, kappa, R, L, D)


tsx, tsr, ts = kranenburg(Q, A, k, kappa, R, L, D)
gamma = 1/ts

a = A/(np.pi*D)

m = 401

x = np.linspace(0,L,100)
r = np.linspace(a,R,100)
t = np.linspace(0,1e5,m)

nx, nr, dt = 500, 500, 10
n_steps = int(t[-1]/dt)

fdres, fdaxs, ndsdx, ndsdr, f = fd_sol(Q, A, k, kappa, R, L, D, nx, nr, n_steps, dt)
t_num = np.arange(0,(n_steps+1)*dt, dt)


result, dsdx, dsdr, ndsdx0, ndsdr0 =  sol(x, r, t, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, gamma)
# result2,dsdx2,dsdr2,ndsdx02,ndsdr02= sol_numf(x, r, t, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, t_num, f, dt)  

############ Analysis of time_scales
x_fd = np.linspace(0,L,nx+1)[:-1]
r_fd = np.linspace(a,R,nr+1)[:-1]


axs = np.append(-x[::-1], r)
s0 = np.append(ssx0(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))
sn = np.append(ssx(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))
s0_fd = np.append(ssx0(x_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr0(r_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))
sn_fd = np.append(ssx(x_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr(r_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))

deltas = sn - s0
# deltas2 = sn- s0
deltas_fd = sn_fd - s0_fd


deltat = np.zeros(len(axs))
# deltat2 = np.zeros(len(axs))
deltat_fd = np.zeros(len(fdaxs))


print('Calculating e-folding...')
for i in range(len(axs)):
    for j in range(len(t)):
        if sn[i] - result[j][i] <= deltas[i]/np.e and deltat[i] == 0:
            deltat[i] = t[j]
print('Calculating e-folding...')
# for i in range(len(axs)):
#     for j in range(len(t)):
#         if sn[i] - result2[j][i] <= deltas2[i]/np.e and deltat2[i] == 0:
#             deltat2[i] = t[j]
print('Calculating e-folding...')
for i in range(len(fdaxs)):
    for j in range(0,len(t_num)-1, int((t[1]-t[0])/dt)):
        if sn_fd[i] - fdres[j][i] <= deltas_fd[i]/np.e and deltat_fd[i] == 0:
            deltat_fd[i] = t_num[j]

print('Calculating e-folding...') 
Tx = 1/eigenvalues_x(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)
Tr = 1/eigenvalues_r(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)
# Tx = 1/(eigenvalues_x(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)-time_scale)
# Tr = 1/(eigenvalues_r(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)-time_scale)

fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(axs, deltat, 'C0-', label='eigen expansion')
# ax.plot(axs, deltat2, 'C2-', label='eigen expansion2')
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
t_num = np.arange(0,(n_steps)*dt, dt)


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
    # ax.plot(axs, result2[i], label = 'eigenfunction expansion')
    ax.plot(fdaxs, fdres[i*n_skips], 'r:', label = 'finite difference')
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
    
    # ax.plot(t[:i], -A*k*dsdx2[:i], 'r--', label=r'$A k s_x(0)$')
    # ax.plot(t[:i], A*kappa*dsdr2[:i],'k--', label=r'$A \kappa s_r(a)$')
    
    ax.hlines(Q*result[i][len(x)-1],0,t[-1], label=r'$Q s(0)$')
    ax.hlines(Q*f[i*n_skips],0,t[-1], label=r'$Q s(0) num$', color='C2')
    
    ax.set_xlim(0, t[-1])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=m-1, interval=10, repeat=False)
plt.show()



# f0 = ssx0(0, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)
# f1 = ssx(0, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)

# fig, ax = plt.subplots(1,1, figsize=(8,8))
# ax.plot(t_num, f, 'C0-', label='finite differences')
# ax.plot(t_num, f1 + (f0-f1)*np.exp(-gamma*t_num), 'C1-', label='kranenburg')
# ax.axhline(f0, color='k', ls='--')
# ax.axhline(f1, color='k', ls='--')
# ax.set_xlim(0, t[-1])
# ax.set_ylim(0, 1)
# ax.set_xlabel(r'$t$')
# ax.set_ylabel(r'$s(0)$')
# ax.legend()
# plt.show()