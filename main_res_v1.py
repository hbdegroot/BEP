#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:51:11 2023

@author: hugo"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import pi
from scipy.special import jv
from scipy.special import yv
from scipy.optimize import root


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


# default = '1'
# Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r = def_func(default)

Q = 100
A = 500
k = 1000
kappa = 1000
L = 10000
R = 10000
D = 2
N_x = 25
N_r = 25

a = A/(np.pi*D)

m = 401
t = np.linspace(0,1e5,m)

nx, nr, dt = 4000, 4000, 100
n_steps = int(t[-1]/dt)

dx = L/(nx+1)
dr = (R-a)/(nr+1)

x = np.linspace(0,L,nx+2)[1:-1]
r = np.linspace(a,R,nr+2)[1:-1]
axs = np.append(-x[::-1], r)

def timescale(Q, A, k, kappa, R, L, D, good_roots_r, nx, nr, dt, n_steps):
    fdres, fdaxs, ndsdx, ndsdr, f = fd_sol(Q, A, k, kappa, R, L, D, nx, nr, n_steps, dt)
    t_num = np.arange(0,(n_steps+1)*dt, dt)
    
    x_fd = np.linspace(0,L,nx+2)[1:-1]
    r_fd = np.linspace(a,R,nr+2)[1:-1]
    
    s0_fd = np.append(ssx0(x_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr0(r_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))
    sn_fd = np.append(ssx(x_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r)[::-1], ssr(r_fd, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r))
    
    deltas_fd = sn_fd - s0_fd
    deltat_fd = np.zeros(len(fdaxs))
    
    print('Calculating e-folding...\n')
    for i in range(len(fdaxs)):
        for j in range(0,len(t_num)-1, max(1,int((t[1]-t[0])/dt))):
            if sn_fd[i] - fdres[j][i] <= deltas_fd[i]/20 and deltat_fd[i] == 0:
                deltat_fd[i] = t_num[j]
    
    return deltat_fd, fdres


def labda0finder(Q, A, k, kappa, R, L, D):
    a = A/(pi*D)
    order = Q/(2*kappa*pi*D)
    def phi_r(r,par):
        return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))
    
    phi_bca = lambda rt: phi_r(a,rt)
    phi_bcR = lambda rt: phi_r(R,rt)
    
    def func_r(rt):
        return phi_bca(rt),phi_bcR(rt)
    

    def n_roots(rt):
         temp = 0
         if abs(phi_r(a,rt))>0.01: return -1
         elif abs(phi_r(R,rt))>0.01: return -1
         for k in range(10,len(r)-20):
             if phi_r(r[k],rt)*phi_r(r[k-1],rt)<0:
                 temp+=1
         return temp
        
    def finder(l0,l1):
        for i in np.linspace(l0,l1,10):
             for j in np.linspace(-10,10,6):
                 rt = root(func_r, x0 =(i,j)).x
                 N = n_roots(rt)
                 #print(rt)
                 if N == 0: return rt[0], rt[1]
                     
                 elif N>0 and rt[0]<l1: return finder(l0/2,rt[0])
        return finder(l1,l1*2)
    return finder(1e-5,1e-4)
        

#kappaspace = np.linspace(kappa/10,kappa*10,5)
varspace = np.array([kappa/10, kappa/2, kappa, kappa*2, kappa*10])

labda_r = []
labda_x = []

deltat = []
fdres = []

for i in range(len(varspace)):
    kappa = varspace[i]
    labda_r.append(labda0finder(Q, A, k, kappa, R, L, D)[0])
    labda_x.append((Q/A)**2/(4*k) + k*(pi/L)**2)
    deltat_fd, fdres = timescale(Q, A, k, kappa, R, L, D, D, nx, nr, dt, n_steps)
    deltat.append(deltat_fd)
    
fig, ax = plt.subplots(1,1, figsize=(8,8))

for i in range(len(varspace)):
    ax.plot(axs, deltat[i], '-', color='C' + str(i), label='kappa = '+str(round(varspace[i],0)))
    ax.hlines(-np.log(0.05)/labda_x[i], -L,0,ls=':', colors='C'+str(i))
    ax.hlines(-np.log(0.05)/labda_r[i], a,R,ls=':', colors='C' + str(i))

ax.set_xlim(-L, R)
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'e-folding $t$')
ax.legend()
plt.show()

#np.savetxt('save_dir/data/20230522_res_def1_200_200_002.csv', fdres)

##########################
####### ANMINATION #######
##########################

# s0_fd = np.append(ssx0(x, Q, A, k, kappa, R, L, D, D, N_x, N_r)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, D, N_x, N_r))
# sn_fd = np.append(ssx(x, Q, A, k, kappa, R, L, D, D, N_x, N_r)[::-1], ssr(r, Q, A, k, kappa, R, L, D, D, N_x, N_r))


# n_skips = int((t[1]-t[0])/dt)
# t_num = np.arange(0,(n_steps)*dt, dt)


# fig, axes = plt.subplots(2,1, figsize=(8,8))

# ax = axes[0]
# ax.plot(axs, s0_fd)
# ax.plot(axs, sn_fd)
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
#     ax.plot(axs, s0_fd, label='initial')
#     ax.plot(axs, sn_fd, label='final')
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
#     ax.plot(axs, s0_fd, label='initial')
#     ax.plot(axs, sn_fd, label='final')
#     ax.plot(axs, fdres[i*n_skips], 'r:', label = 'finite difference')
#     ax.set_xlim(-L, R)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel(r'$x$')
#     ax.set_ylabel(r'$s$')
#     ax.set_title('t=' + str(t[i]))
#     ax.legend()
    
#     ax = axes[1]
#     ax.clear()

#     ax.set_xlim(0, t[-1])
#     ax.set_xlabel(r'$t$')
#     ax.set_ylabel(r'$ds/dx & ds/dr$')
#     ax.legend()
#     return 

# anim = FuncAnimation(fig, animate, frames=m-1, interval=10, repeat=False)
# plt.show()
