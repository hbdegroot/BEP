#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 20:08:10 2023

@author: hugo
"""
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.special import jv
from scipy.special import yv
from scipy.optimize import root
from scipy import integrate


def fd_sol(Q, A, k, kappa, R, L, D, n, m, n_steps, dt):
    print('Finite differences started')    
    a = A/(pi*D)
    
    N = n + m
    dx = L/(n+1)
    dr = (R-a)/(m+1)
    x = np.linspace(0,L,n+2)[1:-1]
    r = np.linspace(a,R,m+2)[1:-1]
    alpha = (kappa/dr) / (kappa/dr + k/dx)
    beta = (k/dx) / (kappa/dr + k/dx)
    
    M = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), k=-1) + np.diag(np.ones(N-1), k=1)
    M[n-1, n-1] = -2 + beta
    M[n-1, n] = alpha
    M[:n,:] = k/(dx**2)*M[:n,:]
    
    M[n, n] = -2 + alpha
    M[n, n-1] = beta
    M[n:,:] = kappa/(dr**2)*M[n:,:]
    
    b = np.zeros(N)
    b[N-1] = kappa/dr**2
    
    def KD(t):
        K = np.diag(np.append(np.ones(n), -np.ones(m)), k=0) + np.diag(np.append(-np.ones(n-1), np.zeros(m)), k=-1) + np.diag(np.append(np.zeros(n), np.ones(m-1)), k=1)
        K[:n,:] = -(Q(t)/A)/(dx)*K[:n,:]
        for j in range(n,N):
            K[j,:] = 1/(a+(j-n+1)*dr) * (kappa-Q(t)/(pi*D)) * 1/(dr)*K[j,:]
        
        d = np.zeros(N)
        d[N-1] = 1/(R-dr) * (kappa-Q(t)/(pi*D)) * 1/(dr)
        return K, d
    
        
    def ssx0(x):
        pwr = Q(0)/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-Q(0)*L/(k*A)))
        c1 = d1*a**pwr
        c2 = -np.exp(-Q(0)*L/(k*A))*c1
        return c1 * np.exp(-Q(0)*x/(k*A)) + c2
    
    def ssr0(r):
        pwr = Q(0)/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-Q(0)*L/(k*A)))
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
        K,d = KD(i*dt)
        P = M + K
        f = b + d
            
        I = np.diag(np.ones(N))
        J = np.linalg.inv(I-dt/2*P)
        T = J @ (I + dt/2*P)
        F = dt*J@f
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res[i] = T @ res[i-1] + F
        
        sx[i] = integrate.trapz(res[i][:n])
        sr[i] = integrate.trapz(res[i][n:]*pi*r*D)
        
    print('Finite differences finished \n')
    return res, sx, sr


def Q(T):
    return 2000 - 1000*np.exp(-T/1e6)

A = 7500
k = 1000
kappa = 200
L = 30000
R = 10000
D = 20
N_x = 25
N_r = 25
c7 = 2


a = A/(np.pi*D)

m = 101
t = np.linspace(0,1e5,m)

nx, nr, dt = 1000, 1000, 1000
n_steps = int(t[-1]/dt)

dx = L/(nx+1)
dr = (R-a)/(nr+1)

x = np.linspace(0,L,nx+2)[1:-1]
r = np.linspace(a,R,nr+2)[1:-1]
axs = np.append(-x[::-1], r)

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


def timescale(Q, A, k, kappa, R, L, D, nx, nr, dt, n_steps):
    fdres, sx, sr = fd_sol(Q, A, k, kappa, R, L, D, nx, nr, n_steps, dt)
    t_num = np.arange(0,(n_steps+1)*dt, dt)
    
    x_fd = np.linspace(0,L,nx+2)[1:-1]
    r_fd = np.linspace(a,R,nr+2)[1:-1]
    
    s0_fd = np.append(ssx(x_fd, Q(0), A, k, kappa, R, L, D)[::-1], ssr(r_fd, Q(0), A, k, kappa, R, L, D))
    sn_fd = np.append(ssx(x_fd, Q(t[-1]), A, k, kappa, R, L, D)[::-1], ssr(r_fd, Q(t[-1]), A, k, kappa, R, L, D))
    
    sx_new = integrate.trapz(ssx(x_fd, Q(t[-1]), A, k, kappa, R, L, D))
    sr_new = integrate.trapz(ssr(r_fd, Q(t[-1]), A, k, kappa, R, L, D)*pi*D*r_fd)
        
    deltas = sn_fd - s0_fd
    deltat = np.zeros(len(axs))
    
    dsx = 0
    dsr = 0
    
    print('Calculating e-folding...\n')
    for i in range(len(axs)):
        for j in range(0,len(t_num)-1, max(1,int((t[1]-t[0])/dt))):
            if abs(sn_fd[i] - fdres[j][i]) <= abs(deltas[i])/20 and deltat[i] == 0:
                deltat[i] = t_num[j]
                
    for j in range(0,len(t_num)-1, max(1,int((t[1]-t[0])/dt))):
            if abs(sx_new - sx[j]) <= abs(sx_new - sx[0])/20 and dsx == 0:
                dsx = t_num[j]
            if abs(sr_new - sr[j]) <= abs(sr_new - sr[0])/20 and dsr == 0:
                dsr = t_num[j]
                print(dsr)
                
    return fdres, deltat, dsx, dsr, sx, sr


def labda0finder(Q, A, k, kappa, R, L, D):
    a = Q(t[-1])/(pi*D)
    order = Q(t[-1])/(2*kappa*pi*D)
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

labda_x = (Q(t[-1])/A)**2/(4*k) + k*(pi/L)**2
labda_r = labda0finder(Q, A, k, kappa, R, L, D)
s0 = np.append(ssx(x, Q(t[0]), A, k, kappa, R, L, D)[::-1], ssr(r, Q(t[0]), A, k, kappa, R, L, D))
sn = np.append(ssx(x, Q(t[-1]), A, k, kappa, R, L, D)[::-1], ssr(r, Q(t[-1]), A, k, kappa, R, L, D))

fdr, delta, dsx, dsr, sx, sr = timescale(Q, A, k, kappa, R, L, D, nx, nr, dt, n_steps)


fig, axes = plt.subplots(1,2, figsize=(8,8))

ax = axes[0]
ax.plot(axs, delta, '-')
ax.hlines(-np.log(0.05)/labda_x, -L,0,ls=':')
ax.hlines(-np.log(0.05)/labda_r, a,R,ls=':')
ax.hlines(dsx, -L,0,ls='--')
ax.hlines(dsr, a,R, ls='--')
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'e-folding $t$')
ax.legend()

ax = axes[1]
ax.plot(axs, delta, '-')
ax.hlines(-np.log(0.05)/labda_x, -L,0,ls=':')
ax.hlines(-np.log(0.05)/labda_r, a,R,ls=':')
ax.hlines(dsx, -L,0,ls='--')
ax.hlines(dsr, a,R, ls='--')
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'e-folding $t$')
ax.legend()

plt.show()

##########################
####### ANMINATION #######
##########################

n_skips = int((t[1]-t[0])/dt)
t_num = np.arange(0,(n_steps)*dt, dt)

fig, axes = plt.subplots(2,1, figsize=(12,12))
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
    ax.plot(axs, s0)
    ax.plot(axs, sn)
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    
    ax = axes[1]
    ax.plot(axs, s0)
    ax.plot(axs, sn)
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax = axes[0]
    ax.clear()
    ax.plot(axs, s0, ls='--')
    ax.plot(axs, np.append(ssx(x, Q(t[i]), A, k, kappa, R, L, D)[::-1], ssr(r, Q(t[i]), A, k, kappa, R, L, D)), ls=':')
    ax.plot(axs, sn, ls='--')
    ax.plot(axs, fdr[i*n_skips], ls='-')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.set_title('t=' + str(t[i]))
    ax.legend()
    
    ax = axes[1]
    ax.clear()
    ax.clear()
    ax.plot(axs, s0, ls='--')
    ax.plot(axs, sn, ls='--')
    ax.plot(axs, fdr[i*n_skips], ls='-')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.set_title('t=' + str(t[i]))
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=m-1, interval=10, repeat=False)
plt.show()
