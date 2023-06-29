#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:09:17 2023

@author: hugo
"""
import numpy as np
from numpy import pi

Q, A, k, kappa, R, L, D = 150, 2000, 400, 400, 10000, 10000, 1

def fd_sol(Q, A, k, kappa, R, L, D, n, m, n_steps, dt):
    print('Finite differences started')
    a = A/(pi*D)

    N = n + m 
    
    dx = L/n
    dr = (R-a)/m
    
    x = np.linspace(0,L,n+1)[:-1]
    r = np.linspace(a,R,m+1)[:-1]
    
    M = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), k=-1) + np.diag(np.ones(N-1), k=1)
    M[n-1,n] = dx/a
    M[n-1,n-1] = -(1+dx/a)
    M[:n,:] = k/(dx**2)*M[:n,:]
    
    M[n,n] = -(1+dr/a)
    M[n,n-1] = dr/a
    M[n:,:] = kappa/(dr**2)*M[n:,:]
    
    b = np.zeros(N)
    b[N-1] = kappa/dr**2
    
    K = np.diag(-np.ones(N-1), k=-1) + np.diag(np.ones(N-1), k=1)
    K[n-1,n] = dx/a
    K[n-1,n-1] = 1-dx/a
    K[:n,:] = -(Q/A)/(2*dx)*K[:n,:]
    
    K[n,n] = -(1-dr/a)
    K[n,n-1] = -dr/a
    for j in range(n,N):
        K[j,:] = 1/r[j-n] * (kappa-Q/(pi*D)) * 1/(2*dr)*K[j,:]
    
    d = np.zeros(N)
    d[N-1] = 1/(R-dr) * (kappa-Q/(pi*D)) * 1/(2*dr)
    
    P = M + K
    f = b + d
    
    I = np.diag(np.ones(N))
    J = np.linalg.inv(I-dt/2*P)
    T = J @ (I + dt/2*P)
    F = dt*J@f

    def ssx0(x):
        pwr = 2*Q/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-2*Q*L/(k*A)))
        c1 = d1*a**pwr
        c2 = -np.exp(-2*Q*L/(k*A))*c1
        return c1 * np.exp(-2*Q*x/(k*A)) + c2
    
    def ssr0(r):
        pwr = 2*Q/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-2*Q*L/(k*A)))
        d2 = 1-d1*R**pwr
        return d1*r**pwr + d2
    
    res = [np.zeros(N) for i in range(n_steps)]
    res[0] = np.append(ssx0(x)[::-1], ssr0(r))
    
    for i in range(1,n_steps):
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res[i] = T @ res[i-1] + F
        #res[i] = res[i-1] + dt*(P @ res[i-1] + f)
        
        axs = np.append(-x[::-1], r)
    
    print('Finite differences finished \n')
    return res, axs

def fv_sol(Q, A, k, kappa, R, L, D, nx, nr, n_steps, dt):
    print('Finite volume started')
    a = A/(pi*D)

    x = np.linspace(0,L,nx+2)[:-1]
    dx = x[1]-x[0]
    r = np.linspace(0,R,nr+2)[1:-1]
    dr = r[1]-r[0]
    
    N = nx + 1 + nr
    
    n = nx
    
    def rj(j):
        #returns r < r_j
        return dr*(j-n-1/2)
    
    M = np.diag(-np.ones(N)) + np.diag(np.ones(N-1), k=-1)
    M[:n,:] = Q/(A*dx)*M[:n,:]
    
    M[n, :] = Q/(pi*dr**2*D/4) * M[n, :]
    
    for j in range(n+1,N):
        M[j, :] = Q/(pi*(rj(j+1)**2-rj(j)**2)/2*D) * M[j, :]
        
    K = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), k=-1) + np.diag(np.ones(N-1), k=1)
    K[:n,:] = k/dx**2*K[:n,:]
    
    K[n, n-1] = k*A/dx / (A*dx/2 + pi*dr**2*D/4)
    K[n, n] = -(k*A/dx + kappa*pi*D/2) / (A*dx/2 + pi*dr**2*D/4)
    K[n, n+1] = kappa*pi/2*D / (A*dx/2 + pi*dr**2*D/4)
    
    for j in range(n+1,N):
        K[j, j-1] = kappa*pi*rj(j)*D / (pi*(rj(j+1)**2-rj(j)**2)/2*D)
        K[j, j] = -kappa*pi*(rj(j)+rj(j+1))*D / (pi*(rj(j+1)**2-rj(j)**2)/2*D)
        if j+1 < N:
            K[j, j+1] = kappa*pi*rj(j+1)*D / (pi*(rj(j+1)**2-rj(j)**2)/2*D)
    
    f = np.zeros(N)
    f[N-1] = kappa*pi*rj(N)*D / (pi*(rj(N)**2-rj(N-1)**2)/2*D)
    
    P = M + K
    
    I = np.diag(np.ones(N))
    J = np.linalg.inv(I-dt/2*P)
    T = J @ (I + dt/2*P)
    F = dt*J@f
    
        
    def ssx(x):
        pwr = Q/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
        c1 = d1*a**pwr
        c2 = -np.exp(-Q*L/(k*A))*c1
        return c1 * np.exp(-Q*x/(k*A)) + c2
    
    def ssr(r):
        pwr = Q/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
        d2 = 1-d1*R**pwr
        return d1*r**pwr + d2
    
    def ssx0(x):
        pwr = 2*Q/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-2*Q*L/(k*A)))
        c1 = d1*a**pwr
        c2 = -np.exp(-2*Q*L/(k*A))*c1
        return c1 * np.exp(-2*Q*x/(k*A)) + c2
    
    def ssr0(r):
        pwr = 2*Q/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-2*Q*L/(k*A)))
        d2 = 1-d1*R**pwr
        return d1*r**pwr + d2
        
    res = [np.zeros(N) for i in range(n_steps)]
    res[0] = np.append(ssx0(x)[::-1], ssr0(r))
    
    for i in range(1,n_steps):
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res[i] = T @ res[i-1] + F
        
    axs = np.append(-x[::-1], r)
    print('Finite volume finished \n')
    return res, axs
