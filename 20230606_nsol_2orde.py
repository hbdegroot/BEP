#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:45:48 2023

@author: hugo
"""

import numpy as np
from numpy import pi
from scipy import integrate

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.special import jv
from scipy.special import yv
from scipy.optimize import root

def fd_sol(Q, A, k, kappa, R, L, D, n, m, n_steps, dt, c7):
    print('Finite differences started')    
    a = A/(pi*D)
    
    N = n + m
    dx = L/(n+1)
    dr = (R-a)/(m+1)
    x = np.linspace(0,L,n+2)[1:-1]
    r = np.linspace(a,R,m+2)[1:-1]
    den = -3*(kappa*dx + k*dr)
    alpha = k*dr/den
    beta = -4*k*dr/den
    gamma = -4*kappa*dx/den
    delta = kappa*dx/den
    
    M = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), k=-1) + np.diag(np.ones(N-1), k=1)
    M[n-1, n-2] = 1 + alpha
    M[n-1, n-1] = -2 + beta
    M[n-1, n] = gamma
    M[n-1, n+1] = delta
    M[:n,:] = k/(dx**2)*M[:n,:]
    
    M[n, n-2] = alpha
    M[n, n-1] = beta
    M[n, n] = -2 + gamma
    M[n, n+1] = 1+delta
    M[n:,:] = kappa/(dr**2)*M[n:,:]
    
    b = np.zeros(N)
    b[N-1] = kappa/dr**2
    
    # K = np.diag(np.append(np.ones(n), -np.ones(m)), k=0) + np.diag(np.append(-np.ones(n-1), np.zeros(m)), k=-1) + np.diag(np.append(np.zeros(n), np.ones(m-1)), k=1)
    # K[:n,:] = -(Q/A)/(dx)*K[:n,:]
    # for j in range(n,N):
    #     K[j,:] = 1/(a+(j-n+1)*dr) * (kappa-Q/(pi*D)) * 1/(dr)*K[j,:]
    
    K = np.diag(-3*np.append(np.ones(n), np.ones(m)), k=0) + np.diag(np.append(4*np.ones(n-1), np.zeros(m)), k=-1) +  np.diag(np.append(-np.ones(n-2), np.zeros(m)), k=-2)
    K = K + np.diag(np.append(np.zeros(n), 4*np.ones(m-1)), k=1) + np.diag(np.append(np.zeros(n), -np.ones(m-2)), k=2) 
    K[:n,:] = (Q/A)/(2*dx)*K[:n,:]
    for j in range(n,N):
        K[j,:] = 1/(a+(j-n+1)*dr) * (kappa-Q/(pi*D)) * 1/(2*dr)*K[j,:]
    
    d = np.zeros(N)
    d[N-1] = 3/(R-dr) * (kappa-Q/(pi*D)) * 1/(2*dr)
    d[N-2] = -1/(R-dr) * (kappa-Q/(pi*D)) * 1/(2*dr)
    
    from scipy.sparse import dia_matrix   
    
    P = M + K
    f = b + d
        
    I = np.diag(np.ones(N))
    J = np.linalg.inv(I-dt/2*P)
    T = J @ (I + dt/2*P)
    F = dt*J@f
        
    def ssx0(x):
        pwr = c7*Q/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-c7*Q*L/(k*A)))
        c1 = d1*a**pwr
        c2 = -np.exp(-c7*Q*L/(k*A))*c1
        return c1 * np.exp(-c7*Q*x/(k*A)) + c2
    
    def ssr0(r):
        pwr = c7*Q/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-c7*Q*L/(k*A)))
        d2 = 1-d1*R**pwr
        return d1*r**pwr + d2
    
    res = [np.zeros(N) for i in range(n_steps)]
    res[0] = np.append(ssx0(x)[::-1], ssr0(r))
    
    
    sx = np.zeros(n_steps)
    sx[0] = integrate.trapz(ssx0(x))
    sr = np.zeros(n_steps)
    sr[0] = integrate.trapz(ssr0(r)*pi*r*D)
    
    #T = dia_matrix(T)
    
    for i in range(1,n_steps):
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res[i] = T @ res[i-1] + F
        
        sx[i] = integrate.trapz(res[i][:n])
        sr[i] = integrate.trapz(res[i][n:]*pi*r*D)
        
    print('Finite differences finished \n')
    return res, sx, sr

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

m = 401
t = np.linspace(0,2e6,m)

nx, nr, dt = 1000, 10000, 1000
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


def timescale(Q, A, k, kappa, R, L, D, nx, nr, dt, n_steps, c7):
    fdres, sx, sr = fd_sol(Q, A, k, kappa, R, L, D, nx, nr, n_steps, dt, c7)
    t_num = np.arange(0,(n_steps+1)*dt, dt)
    
    x_fd = np.linspace(0,L,nx+2)[1:-1]
    r_fd = np.linspace(a,R,nr+2)[1:-1]
    
    s0_fd = np.append(ssx0(x_fd, Q, A, k, kappa, R, L, D, c7)[::-1], ssr0(r_fd, Q, A, k, kappa, R, L, D, c7))
    sn_fd = np.append(ssx(x_fd, Q, A, k, kappa, R, L, D)[::-1], ssr(r_fd, Q, A, k, kappa, R, L, D))
    
    sx_new = integrate.trapz(ssx(x_fd, Q, A, k, kappa, R, L, D))
    sr_new = integrate.trapz(ssr(r_fd, Q, A, k, kappa, R, L, D)*pi*D*r_fd)
        
    deltas = sn_fd - s0_fd
    deltat = np.zeros(len(axs))
    
    dsx = 0
    dsr = 0
    
    print('Calculating e-folding...\n')
    for i in range(len(axs)):
        for j in range(0,len(t_num)-1):
            if abs(sn_fd[i] - fdres[j][i]) <= abs(deltas[i])/20 and deltat[i] == 0:
                deltat[i] = t_num[j]
                
    for j in range(0,len(t_num)-1):
            if abs(sx_new - sx[j]) <= abs(sx_new - sx[0])/20 and dsx == 0:
                dsx = t_num[j]
            if abs(sr_new - sr[j]) <= abs(sr_new - sr[0])/20 and dsr == 0:
                dsr = t_num[j]
                print(dsr)
                
    return fdres, deltat, dsx, dsr, sx, sr


def labda0finder(Q, A, k, kappa, R, L, D):
    a = Q/(pi*D)
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

#varspace = np.array([2,0.5])        
varspaceQ = np.array([0.5,2])
varspace = np.array([kappa/10])  

labda_r = []
labda_x = []
delta = []
fd = []
tx, tr = [], []
sx_arr, sr_arr = [], []
sn, s0 = [], []

labda_r2 = []
labda_x2 = []
delta2 = []
fd2 = []
tx2, tr2 = [], []
sx_arr2, sr_arr2 = [], []
sn2, s02 = [], []

for i in range(len(varspace)):
    kappa = varspace[i]
    c7 = varspaceQ[0]
    labda_r.append(labda0finder(Q, A, k, kappa, R, L, D)[0])
    labda_x.append((Q/A)**2/(4*k) + k*(pi/L)**2)
    fdr, deltat_fd, dsx, dsr, sx, sr = timescale(Q, A, k, kappa, R, L, D, nx, nr, dt, n_steps, c7)
    delta.append(deltat_fd)
    tx.append(dsx)
    tr.append(dsr)
    fd.append(fdr)
    sx_arr.append(sx)
    sr_arr.append(sr)
    s0.append(np.append(ssx0(x, Q, A, k, kappa, R, L, D, c7)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, c7)))
    sn.append(np.append(ssx(x, Q, A, k, kappa, R, L, D)[::-1], ssr(r, Q, A, k, kappa, R, L, D)))
    
    c7 = varspaceQ[1]
    labda_r2.append(labda0finder(Q, A, k, kappa, R, L, D)[0])
    labda_x2.append((Q/A)**2/(4*k) + k*(pi/L)**2)
    fdr2, deltat_fd2, dsx2, dsr2, sx2, sr2 = timescale(Q, A, k, kappa, R, L, D, nx, nr, dt, n_steps, c7)
    delta2.append(deltat_fd2)
    tx2.append(dsx2)
    tr2.append(dsr2)
    fd2.append(fdr2)
    sx_arr2.append(sx2)
    sr_arr2.append(sr2)
    s02.append(np.append(ssx0(x, Q, A, k, kappa, R, L, D, c7)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, c7)))
    sn2.append(np.append(ssx(x, Q, A, k, kappa, R, L, D)[::-1], ssr(r, Q, A, k, kappa, R, L, D)))


fig, axes = plt.subplots(1,2, figsize=(8,8))

ax = axes[0]
for i in range(len(varspace)):
    ax.plot(axs, delta[i], '-', color='C' + str(i), label='$\kappa$ = '+str(round(varspace[i],1)))
    ax.hlines(-np.log(0.05)/labda_x[i], -L,0,ls=':', colors='C'+str(i))
    ax.hlines(-np.log(0.05)/labda_r[i], a,R,ls=':', colors='C' + str(i))
    ax.hlines(tx[i], -L,0,ls='--', colors='C' + str(i))
    ax.hlines(tr[i], a,R, ls='--', colors='C' + str(i))
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'e-folding $t$')
ax.legend()

ax = axes[1]
for i in range(len(varspace)):
    ax.plot(axs, delta2[i], '-', color='C' + str(i), label='$\kappa$ = '+str(round(varspace[i],1)))
    ax.hlines(-np.log(0.05)/labda_x2[i], -L,0,ls=':', colors='C'+str(i))
    ax.hlines(-np.log(0.05)/labda_r2[i], a,R,ls=':', colors='C' + str(i))
    ax.hlines(tx2[i], -L,0,ls='--', colors='C' + str(i))
    ax.hlines(tr2[i], a,R, ls='--', colors='C' + str(i))
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'e-folding $t$')
ax.legend()
plt.show()


T =  np.arange(0,(n_steps)*dt, dt)

# fig, ax = plt.subplots(1,1, figsize=(8,8))
# for i in range(len(varspace)):
#     ax.plot(T, sx_arr[i])
#     ax.plot(T, sx_arr[i])

# ax.set_xlim(0, t[-1])
# ax.set_xlabel(r'$-x & r$')
# ax.set_ylabel(r'e-folding $t$')
# ax.legend()
# plt.show()

#np.savetxt('save_dir/data/20230522_res_def1_200_200_002.csv', fdres)

##########################
####### ANMINATION #######
##########################

n_skips = int((t[1]-t[0])/dt)
t_num = np.arange(0,(n_steps)*dt, dt)

fig, axes = plt.subplots(2,1, figsize=(12,12))
ax = axes[0]
for i in range(len(varspace)):
    ax.plot(axs, s0[i], color='C' + str(i))
    ax.plot(axs, sn[i], color='C' + str(i))
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
    for i in range(len(varspace)):
        ax.plot(axs, s0[i])
        ax.plot(axs, sn[i])
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    
    ax = axes[1]
    for i in range(len(varspace)):
        ax.plot(axs, s02[i])
        ax.plot(axs, sn2[i])
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax = axes[0]
    ax.clear()
    for j in range(len(varspace)):
        ax.plot(axs, s0[j], ls='--', color='C' + str(j))
        ax.plot(axs, sn[j], ls='--', color='C' + str(j))
        ax.plot(axs, fd[j][i*n_skips], ls='-', color='C' + str(j), label = 'var = ' + str(varspace[j]))
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.set_title('t=' + str(t[i]))
    ax.legend()
    
    ax = axes[1]
    ax.clear()
    for j in range(len(varspace)):
        ax.plot(axs, s02[j], ls='--', color='C' + str(j))
        ax.plot(axs, sn2[j], ls='--', color='C' + str(j))
        ax.plot(axs, fd2[j][i*n_skips], ls='-', color='C' + str(j), label = 'var = ' + str(varspace[j]))
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.set_title('t=' + str(t[i]))
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=m-1, interval=10, repeat=False)
plt.show()


# for i in range(len(varspace)):
#     run = 'kappa' + str(varspace[i]) + 'Q' + str(5)
#     path = 'save_dir/data/20230606_2de_orde/'
#     np.savetxt(path+run+'labda_r', labda_r)
#     np.savetxt(path+run+'labda_x', labda_x)
#     np.savetxt(path+run+'delta', delta)
#     np.savetxt(path+run+'tx', tx)
#     np.savetxt(path+run+'tr', tr)
#     #np.savetxt(path+run+'fd', fd[i])
#     np.savetxt(path+run+'sx_arr', sx_arr[i])
#     np.savetxt(path+run+'sr_arr', sr_arr[i])
#     np.savetxt(path+run+'s0', s0[i])
#     np.savetxt(path+run+'sn', sn[i])
    
#     run = 'kappa' + str(varspace[i]) + 'Q' + str(2)
#     path = 'save_dir/data/20230606_2de_orde/'
#     np.savetxt(path+run+'labda_r', labda_r2)
#     np.savetxt(path+run+'labda_x', labda_x2)
#     np.savetxt(path+run+'delta', delta2)
#     np.savetxt(path+run+'tx', tx2)
#     np.savetxt(path+run+'tr', tr2)
#     #np.savetxt(path+run+'fd', fd2[i])
#     np.savetxt(path+run+'sx_arr', sx_arr2[i])
#     np.savetxt(path+run+'sr_arr', sr_arr2[i])
#     np.savetxt(path+run+'s0', s02[i])
#     np.savetxt(path+run+'sn', sn2[i])