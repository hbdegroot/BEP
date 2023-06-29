#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:38:39 2023

@author: hugo
"""

import numpy as np
from numpy import pi
from scipy import integrate

Q, A, k, kappa, R, L, D = 150, 2000, 400, 400, 10000, 10000, 1

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
    
    K = np.diag(np.append(np.ones(n), -np.ones(m)), k=0) + np.diag(np.append(-np.ones(n-1), np.zeros(m)), k=-1) + np.diag(np.append(np.zeros(n), np.ones(m-1)), k=1)
    K[:n,:] = -(Q/A)/(dx)*K[:n,:]
    for j in range(n,N):
        K[j,:] = 1/(a+(j-n+1)*dr) * (kappa-Q/(pi*D)) * 1/(dr)*K[j,:]
    
    d = np.zeros(N)
    d[N-1] = 1/(R-dr) * (kappa-Q/(pi*D)) * 1/(dr)
    
    from scipy.sparse import dia_matrix   
    
    P = M + K
    f = b + d
        
    I = np.diag(np.ones(N))
    J = np.linalg.inv(I-dt/2*P)
    T = J @ (I + dt/2*P)
    F = dt*J@f
    
    c7 = 0.5
    
    def ssx0(x):
        pwr = c7*Q/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-c7*Q*L/(k*A)))
        c1 = d1*a**pwr
        c2 = -np.exp(-c7*Q*L/(k*A))*c1
        return c1 * np.exp(-2*Q*x/(k*A)) + c2
    
    def ssr0(r):
        pwr = c7*Q/(kappa*pi*D)
        d1 = 1/(R**pwr -a**pwr*np.exp(-c7*Q*L/(k*A)))
        d2 = 1-d1*R**pwr
        return d1*r**pwr + d2
    
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
    
    res = [np.zeros(N) for i in range(n_steps)]
    res[0] = np.append(ssx0(x)[::-1], ssr0(r))
    
    dsdx = np.zeros(n_steps)
    dsdr = np.zeros(n_steps)
    
    f = np.zeros(n_steps)
    f[0] = alpha*res[0][n] + beta*res[0][n-1]
    
    T = dia_matrix(T)
    
    for i in range(1,n_steps):
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res[i] = T @ res[i-1] + F
        
        # dsdx[i] =  (-alpha*res[i][n] + (1-beta)*res[i][n-1])/dx
        # dsdr[i] =  (-beta*res[i][n-1] + (1-alpha)*res[i][n])/dr
        # f[i] = alpha*res[i][n] + beta*res[i][n-1]
        #res[i] = res[i-1] + dt*(P @ res[i-1] + f)
    
    axs = np.append(-x[::-1], r)
        
    print('Finite differences finished \n')
    return res, axs, dsdx, dsdr, f

def kranenburg(Q, A, k, kappa, R, L, D):
    a = A/(pi*D)
    def s_Qx(x):
        o = Q/(kappa*pi*D)
        u = Q/(k*A)
        
        n = (R/a)**o - np.exp(-u*L)
        T1 = n * ( -x/(k*A)*np.exp(-u*x) + L/(k*A)*np.exp(-u*L) )
        T2 = ( np.exp(-u*x) - np.exp(-u*L) ) * ( 1/(kappa*pi*D)*np.log(R/a)*(R/a)**o )
        return (T1 - T2)/n**2 
    
    def s_Qr(r):
        o = Q/(kappa*pi*D)
        u = Q/(k*A)
        
        n = R**o - a**o * np.exp(-u*L)
        T1 = n * 1/(kappa*pi*D)*( np.log(r)*r**o - np.log(R)*R**o )
        T2 = ( r**o  - R**o ) * ( 1/(kappa*pi*D) * ( np.log(R)*R**o - np.log(a)*a**o * np.exp(-u*L) ) + L/(k*A)*a**o * np.exp(-u*L) )
        return (T1 - T2)/n**2 
    
    
    tsx = integrate.quad(lambda ksi: np.abs(A*s_Qx(ksi)), 0, L)[0]
    tsr = integrate.quad(lambda psi: np.abs(pi*psi*D*s_Qr(psi)), a, R)[0]
    return tsx, tsr, tsx + tsr