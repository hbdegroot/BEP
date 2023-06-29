#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:34:35 2023

@author: hugo
"""
import numpy as np
from numpy import pi
from scipy import integrate

# Q, A, k, kappa, R, L, D = 150, 2000, 400, 400, 10000, 10000, 1

def fd_sol(Q, A, k, kappa, R, L, D, n, m, n_steps, dt, c7):
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

def fd_sol_faster(Q, A, k, kappa, R, L, D, n, m, n_steps, dt, c7):
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
    
    res = np.append(ssx0(x)[::-1], ssr0(r))
    
    #T = dia_matrix(T)
    
    for i in range(1,n_steps):
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res = T @ res + F
        
    print('Finite differences finished \n')
    return res

def fd_sol2(Q, A, k_t, kappa_t, R, L, D, n, m, n_steps, dt, ssx0, ssr0):
    print('Finite differences started')    
    a = A/(pi*D)
    
    N = n + m
    dx = L/(n+1)
    dr = (R-a)/(m+1)
    x = np.linspace(0,L,n+2)[1:-1]
    r = np.linspace(a,R,m+2)[1:-1]
    
    def KD(t):
        k = k_t(t)
        kappa = kappa_t(t)
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
        K[:n,:] = -(Q(t)/A)/(dx)*K[:n,:]
        for j in range(n,N):
            K[j,:] = 1/(a+(j-n+1)*dr) * (kappa-Q(t)/(pi*D)) * 1/(dr)*K[j,:]
        
        d = np.zeros(N)
        d[N-1] = 1/(R-dr) * (kappa-Q(t)/(pi*D)) * 1/(dr)
        return M,K,b,d
    
        
    
    res = [np.zeros(N) for i in range(n_steps)]
    res[0] = np.append(ssx0[::-1], ssr0)
    
    
    sx = np.zeros(n_steps)
    sx[0] = integrate.trapz(ssx0)
    sr = np.zeros(n_steps)
    sr[0] = integrate.trapz(ssr0*pi*r*D)
    
    #T = dia_matrix(T)
    
    for i in range(1,n_steps):
        M,K,b,d = KD(i*dt)
        M2,K2,b2,d2 = KD((i+1)*dt)
        P = M + K
        P2 = M2 + K2
        f = b + d
        f2 = b2 + d2
            
        I = np.diag(np.ones(N))
        J = np.linalg.inv(I-dt/2*P2)
        T = J @ (I + dt/2*P)
        F = dt* J @ (f+f2)/2
        
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res[i] = T @ res[i-1] + F
        
        #sx[i] = integrate.trapz(res[i][:n])
        #sr[i] = integrate.trapz(res[i][n:]*pi*r*D)
        
    print('Finite differences finished \n')
    return res, sx, sr

def fd_sol_plus_initial(Q, A, k, kappa, R, L, D, n, m, n_steps, dt, ssx0, ssr0):
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
    
        
    
    res = [np.zeros(N) for i in range(n_steps)]
    res[0] = np.append(ssx0[::-1], ssr0)
    
    
    sx = np.zeros(n_steps)
    sx[0] = integrate.trapz(ssx0)
    sr = np.zeros(n_steps)
    sr[0] = integrate.trapz(ssr0*pi*r*D)
    

    P = M + K
    f = b + d
        
    I = np.diag(np.ones(N))
    J = np.linalg.inv(I-dt/2*P)
    T = J @ (I + dt/2*P)
    F = dt*J@f    
    for i in range(1,n_steps):
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res[i] = T @ res[i-1] + F
        
        sx[i] = integrate.trapz(res[i][:n])
        sr[i] = integrate.trapz(res[i][n:]*pi*r*D)
        
    print('Finite differences finished \n')
    return res, sx, sr

def fd_sol2_fw(Q, A, k_t, kappa, R, L, D, n, m, n_steps, dt, ssx0, ssr0, detail=1):
    print('Finite differences started')    
    a = A/(pi*D)
    
    N = n + m
    dx = L/(n+1)
    dr = (R-a)/(m+1)
    x = np.linspace(0,L,n+2)[1:-1]
    r = np.linspace(a,R,m+2)[1:-1]
    
    def KD(t):
        k = k_t(t)
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
        K[:n,:] = -(Q(t)/A)/(dx)*K[:n,:]
        for j in range(n,N):
            K[j,:] = 1/(a+(j-n+1)*dr) * (kappa-Q(t)/(pi*D)) * 1/(dr)*K[j,:]
        
        d = np.zeros(N)
        d[N-1] = 1/(R-dr) * (kappa-Q(t)/(pi*D)) * 1/(dr)
        return M,K,b,d
    
        
    
    res = [np.zeros(N) for i in range(n_steps)]
    res[0] = np.append(ssx0[::-1], ssr0)
    temp = res[0]
    
    
    sx = np.zeros(n_steps)
    sx[0] = integrate.trapz(ssx0)
    sr = np.zeros(n_steps)
    sr[0] = integrate.trapz(ssr0*pi*r*D)
    
    #T = dia_matrix(T)
    
    for i in range(1,n_steps):
        if i%10 == 1:
            M,K,b,d = KD(i*dt)
            P = M + K
            f = b + d
                
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        temp = temp + dt * (P @ temp + f)
        if i%detail == 0:
            res[i//detail] = temp
            sx[i] = integrate.trapz(res[i][:n])
            sr[i] = integrate.trapz(res[i][n:]*pi*r*D)
        
    print('Finite differences finished \n')
    return res, sx, sr


def fd_sol_transport(Q, A, k, kappa, R, L, D, n, m, n_steps, dt, c7):
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
    
    
    T_A = [np.zeros(N) for i in range(n_steps)]
    T_D = [np.zeros(N) for i in range(n_steps)]
    
    T_D[0][1:n] = -A*k* (res[0][:n-1] - res[0][1:n])/dx
    T_D[0][n:-1] = pi*r[:-1]*D*kappa* (res[0][n+1:] - res[0][n:-1])/dr
    
    T_A[0][:n] = Q*res[0][:n] 
    T_A[0][n:] = Q*res[0][n:] 

    for i in range(1,n_steps):
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res[i] = T @ res[i-1] + F
        
        T_D[i][1:n] = -A*k* (res[i][:n-1] - res[i][1:n])/dx
        T_D[i][n:-1] = pi*r[:-1]*D*kappa* (res[i][n+1:] - res[i][n:-1])/dr
        
        T_A[i][:n] = Q*res[i][:n] 
        T_A[i][n:] = Q*res[i][n:] 
        
    print('Finite differences finished \n')
    return res, T_A, T_D

def fd_sol_transport_sea(Q, A, kappa, R, D, m, n_steps, dt, c7):
    print('Finite differences started')    
    a = A/(pi*D)
    N = m
    dr = (R-a)/(m+1)
    r = np.linspace(a,R,m+2)[1:-1]
    
    M = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), k=-1) + np.diag(np.ones(N-1), k=1)
    M = kappa/(dr**2)*M
    
    b = np.zeros(N)
    b[0] = 0.5*kappa/dr**2
    b[N-1] = kappa/dr**2
    
    K = np.diag(-np.ones(N), k=0)  + np.diag(np.ones(N-1), k=1)
    for j in range(N):
        K[j,:] = 1/(a+(j+1)*dr) * (kappa-Q/(pi*D)) * 1/(dr)*K[j,:]
    
    d = np.zeros(N)
    d[N-1] = 1/(R-dr) * (kappa-Q/(pi*D)) * 1/(dr)
        
    P = M + K
    f = b + d
        
    I = np.diag(np.ones(N))
    J = np.linalg.inv(I-dt/2*P)
    T = J @ (I + dt/2*P)
    F = dt*J@f
        
    def ssr0(r):
        pwr = c7*Q/(kappa*pi*D)
        d1 = (0.5-1)/(a**pwr - R**pwr)
        return d1*(r**pwr - R**pwr) + 1
    
    res = [np.zeros(N) for i in range(n_steps)]
    res[0] = ssr0(r)
    
    T_A = [np.zeros(N) for i in range(n_steps)]
    T_D = [np.zeros(N) for i in range(n_steps)]
    
    T_D[0][:-1] = pi*r[:-1]*D*kappa* (res[0][1:] - res[0][:-1])/dr
    T_A[0] = Q*res[0]
    
    for i in range(1,n_steps):
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res[i] = T @ res[i-1] + F
        
        T_D[i][:-1] = pi*r[:-1]*D*kappa*(res[i][1:] - res[i][:-1])/dr
        
        T_A[i] = Q*res[i]
        
    print('Finite differences finished \n')
    return res, T_A, T_D


def fd_sol_horde(Q, A, k, kappa, R, L, D, n, m, n_steps, dt, c7):
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
    
    res = np.append(ssx0(x)[::-1], ssr0(r))
    
    
    
    for i in range(1,n_steps):
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res = T @ res + F

    print('Finite differences finished \n')
    return res


def fd_sol_horde2(Q, A, k, kappa, R, L, D, n, m, n_steps, dt, c7):
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
    
    for i in range(1,n_steps):
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res[i] = T @ res[i-1] + F
        
        sx[i] = integrate.trapz(res[i][:n])
        sr[i] = integrate.trapz(res[i][n:]*pi*r*D)
        
    print('Finite differences finished \n')
    return res, sx, sr