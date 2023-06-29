#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:37:07 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import integrate

from scipy.special import jv
from scipy.special import yv
from scipy.optimize import root

pi = np.pi

N = 30

f = 0.8
perc = 1/np.e

rho = np.linspace(0,1,500)
tau = np.linspace(0,0.4,1000)
def dmlts(P,q,m):
    def ss(rho):
        temp = ( (rho+q)**P - q**P )/( (1+q)**P - q**P ) 
        return temp*(1-f) + f
    
    def h1(rho):
        temp = ( (rho+q)**(P/m) - q**(P/m) )/( (1+q)**(P/m) - q**(P/m) ) 
        return temp*(1-f) + f

    def h2(rho):
        temp = ( (rho+q)**(P*m) - q**(P*m) )/( (1+q)**(P*m) - q**(P*m) ) 
        return temp*(1-f) + f
    
    def phi_r(rho,P,q,par):
        return (rho+q)**(P/2)*jv(P/2, np.sqrt(par[0])*(rho+q)) + par[1]*(rho+q)**(P/2)*yv(P/2, np.sqrt(par[0])*(rho+q))
      
    r = np.linspace(0,1,500)
    
    def good_root_finder(N,P,q):
        phi_bc0 = lambda rt: phi_r(0,P,q,rt)
        phi_bc1 = lambda rt: phi_r(1,P,q,rt)
             
        def func_r(rt):
            return phi_bc0(rt),phi_bc1(rt)
            
        def n_roots(rt):
            temp = 0
            if abs(phi_r(0,P,q,rt))>0.01: return -1
            elif abs(phi_r(1,P,q,rt))>0.01: return -1
            for k in range(5,len(r)-5):
                if phi_r(r[k],P,q,rt)*phi_r(r[k-1],P,q,rt)<0:
                    temp+=1
            return temp
        
        def finder(l0,l1,n):
            for i in np.linspace(l0,l1,10):
                 for j in np.linspace(-10,10,6):
                     rt = root(func_r, x0 =(i,j)).x
                     M = n_roots(rt)
                     #print(rt)
                     if M == n: return rt[0], rt[1]
                         
                     elif M>n and rt[0]<l1: return finder(l0/2,rt[0],n)
            return finder(l1,l1*2,n)
        
        roots = [np.zeros(2) for n in range(N)]
        for n in range(N):
            roots[n] = finder(roots[n-1][0],roots[n-1][0]*1.5+1e-1,n)                                                                                      
        return roots
    
    #print('Looking for some roots...')
    good_roots = good_root_finder(N,P,q)
    #print('They\'re  here!\nContinue integrating...\n')
    sign = [np.sign(phi_r(0.01, P, q, good_roots[n]))/np.sqrt(integrate.quad(lambda r: phi_r(r, P, q, good_roots[n])**2, 0, 1)[0]) for n in range(N)]
    
    def phi_n(r,n):
        return phi_r(r, P, q, good_roots[n]) * sign[n]
    
    def labda_n(n):
        return good_roots[n][0]
    
    def inner(n,m):
        return integrate.quad(lambda r: phi_n(r,m)*phi_n(r,n), 0, 1)[0]
    
    G = np.array([[inner(n,m) for n in range(N)] for m in range(N)])
    
    labda = np.array([labda_n(n) for n in range(N)])
    
    inv = np.linalg.inv(G)
    def solf(h):
        I1 = [sum([inv[n,j]*integrate.quad(lambda rho: (h(rho)-f-rho*(1-f))*phi_n(rho,j), 0, 1)[0] for j in range(N)]) for n in range(N)]
        I2 = [sum([inv[n,j]*integrate.quad(lambda rho: (1-P)/(rho+q)*(1-f)*phi_n(rho,j), 0, 1)[0] for j in range(N)]) for n in range(N)]
        
        tempi1 = np.array([np.exp(-labda[n]*tau) for n in range(N)])
        tempi2 = np.array([(1-np.exp(-labda[n]*tau))/labda[n] for n in range(N)])
        
        T = np.zeros((len(tau), N))
        
        for i in range(len(tau)):
            T1 = tempi1[:,i] * I1
            T2 = tempi2[:,i] * I2
            T[i,:] = T1 + T2
          
        def ksi(rho):
            return f + (1-f)*rho
        
        KSI = np.zeros((len(tau), len(rho)))
        for i in range(len(tau)):
            for j in range(len(rho)):
                KSI[i,j] = ksi(rho[j])
        
        sol = T @ np.array([phi_n(rho,n) for n in range(N)]) + KSI
    
        s0 = h(rho)
        s9 = ss(rho)
    
        deltas = s9 - s0
        deltat = np.zeros(len(rho))
        if np.sum(deltas)  < 0:
            for i in range(len(rho)):
                for j in range(len(tau)):
                    if s9[i] - sol[j][i] >= deltas[i]*perc and deltat[i] == 0:
                        deltat[i] = tau[j]
        else:
            for i in range(len(rho)):
                for j in range(len(tau)):
                    if s9[i] - sol[j][i] <= deltas[i]*perc and deltat[i] == 0:
                        deltat[i] = tau[j]
                        
        dt_min = np.min(deltat[10:-10])
        dt_max = np.max(deltat[10:-10])

        dstot = 0
        Stot0 = integrate.quad(lambda rho: h(rho)*pi*(rho+q), 0, 1)[0]
        Stot9 = integrate.quad(lambda rho: ss(rho)*pi*(rho+q), 0, 1)[0]
        for j in range(len(tau)):
            if abs(Stot9 - integrate.trapz(sol[j]*pi*(rho+q), rho)) < perc*abs(Stot9 - Stot0) and dstot ==0:
                dstot = tau[j]
        return dstot, dt_min, dt_max, deltat
    
    dstot1, dt_min1, dt_max1, deltat1 = solf(h1)
    dstot2, dt_min2, dt_max2, deltat2 = solf(h2)
                
    return dstot1, dstot2, dt_min1, dt_min2, dt_max1, dt_max2, -np.log(perc)/labda[0], deltat1, deltat2


q = 0.02
P = np.linspace(0.01,0.5,11)

mspace = np.linspace(1.5,10,5 )
ds1, ds2 = [np.zeros(len(P)) for i in range(len(mspace))], [np.zeros(len(P)) for i in range(len(mspace))]
dt1, dt2 = [[np.zeros(len(rho)) for j in range(len(P))] for i in range(len(mspace))], [[np.zeros(len(rho)) for j in range(len(P))] for i in range(len(mspace))]
dtmin1, dtmin2 = [np.zeros(len(P)) for i in range(len(mspace))], [np.zeros(len(P)) for i in range(len(mspace))]
dtmax1, dtmax2 = [np.zeros(len(P)) for i in range(len(mspace))], [np.zeros(len(P)) for i in range(len(mspace))]
evts = [np.zeros(len(P)) for i in range(len(mspace))]


for j in range(len(mspace)):
    print('Calculation', j, '/', len(mspace), 'started')
    for i in range(len(P)):
        ds1[j][i], ds2[j][i], dtmin1[j][i], dtmin2[j][i], dtmax1[j][i], dtmax2[j][i], evts[j][i], dt1[j][i], dt2[j][i] = dmlts(P[i], q, mspace[j])

yerr1 = [np.zeros((2,len(P))) for j in range(len(mspace))]
yerr2 = [np.zeros((2,len(P))) for j in range(len(mspace))]
for j in range(len(mspace)):
    for i in range(len(P)):
        yerr1[j][0,i], yerr1[j][1,i] = dtmin1[j][i], dtmax1[j][i]
        yerr2[j][0,i], yerr2[j][1,i] = dtmin2[j][i], dtmax2[j][i]
        
ms1 = [(dtmin1[j]+ dtmax1[j])**2*1e4 for j in range(len(mspace))]
ms2 = [(dtmin2[j]+ dtmax2[j])**2*1e4 for j in range(len(mspace))]

c = []
for j in range(len(mspace)):
    c.append([mspace[j]]*len(P))
    
    
fig, ax = plt.subplots(1,1, figsize=(8,8))
im = ax.scatter([P for j in range(len(mspace))], [ds1[j] for j in range(len(mspace))], s=[ms1[j] for j in range(len(mspace))], c=c , cmap='viridis', marker='v', label='decreasing $s$')
ax.scatter([P for j in range(len(mspace))], [ds2[j] for j in range(len(mspace))], s=[ms2[j] for j in range(len(mspace))], c=c , cmap='viridis', marker='^', label='increasing $s$')
ax.plot(P, evts[j], 'k-', label=r'$1/\Lambda_1$')
ax.set_xlim(0,P[-1]+0.02)
ax.set_ylim(0.10,0.17)
ax.set_xlabel(r'$\mathrm{P}$')
ax.set_ylabel(r'$\mathcal{T}_{ADJ}$')
plt.colorbar(im, ax=ax, label=r'$m$')
lgnd = ax.legend(loc="upper right", scatterpoints=1, fontsize=10)
lgnd.legendHandles[1]._sizes = [40]
lgnd.legendHandles[2]._sizes = [40]
#ax.legend()
plt.show()

sym = ['.', '*', '^', 'v', 's']

fig, axes = plt.subplots(1,2, figsize=(8,8))
ax = axes[0]
for j in range(len(P)):
    for i in range(len(mspace)):
        ax.scatter(rho, dt1[i][j], marker=sym[i], c='C'+str(j))
        
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\mathcal{T}_{ADJ}$')

ax = axes[1]
for j in range(len(P)):
    for i in range(len(mspace)):
        ax.scatter(rho, dt2[i][j], marker=sym[i], c='C'+str(j))
        
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\mathcal{T}_{ADJ}$')
#ax.legend()
plt.show()
