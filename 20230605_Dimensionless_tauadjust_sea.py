#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:54:59 2023

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
f = 0.5
perc = 1/np.e

rho = np.linspace(0,1,1000)
tau = np.linspace(0,100,200)

N = 20

def goodrootfinder(N_r, l0, lmax, fine, P,q):
    order = P/2
    def phi_r(rho,par):
        return (rho+q)**(order)*jv(order, np.sqrt(par[0])*(rho+q)) + par[1]*(rho+q)**(order)*yv(order, np.sqrt(par[0])*(rho+q))
    
    phi_bc0 = lambda rt: phi_r(0,rt)
    phi_bc1 = lambda rt: phi_r(1,rt)
    
    def func_r(rt):
        return phi_bc0(rt),phi_bc1(rt)
    
    rho = np.linspace(0,1,500)
    def n_roots(rt):
        temp = 0
        if abs(phi_r(0,rt))>0.01: return -1
        elif abs(phi_r(1,rt))>0.01: return -1
        for j in range(10,len(rho)-20):
            if phi_r(rho[j],rt)*phi_r(rho[j-1],rt)<0:
                temp+=1
        return temp
    
    roots = [[0,0] for i in range(N_r)]
    
    for i in np.linspace(l0,lmax,fine):
        for j in np.linspace(-10,10,4):
            rt = root(func_r, x0 =(i,j)).x
            N = n_roots(rt)
            #print(rt)
            if N<N_r:
                if N+1>0 and roots[N][0] == 0:
                    roots[N] = rt
        
    mistake = False
    for i in range(len(roots)):
        if roots[i][0] == 0:
            print('Warning: root 0 detected @ N =', i)
            mistake = True
    if mistake:
        for i in range(len(roots)):
            print(i, roots[i])
        raise ValueError('root 0 encountered')
    return roots


def dmlts(P,q, multiplier=2):
    def ss(rho):
        temp = (1+q)**P
        return (1-f)/(temp-q**P) *( (rho+q)**P - temp ) + 1
    
    def h(rho):
        P2 = P/multiplier
        temp = (1+q)**P2
        return (1-f)/(temp-q**P2) *( (rho+q)**P2 - temp ) + 1
    
    good_roots = goodrootfinder(N, 1, 4000, 2000, P,q)
    print('All set, continue integrating \n')
    
    order = P/2
    def phi_r(rho,par):
        return (rho+q)**(order)*jv(order, np.sqrt(par[0])*(rho+q)) + par[1]*(rho+q)**(order)*yv(order, np.sqrt(par[0])*(rho+q))
    
    def phi_n(rho,n):
        return phi_r(rho, good_roots[n])
    
    def labda_n(n):
        return good_roots[n][0]
    
    def inner(n,m):
        return integrate.quad(lambda xi: phi_n(xi,m)*phi_n(xi,n), 0, 1)[0]
    
    G = np.array([[inner(n,m) for n in range(N)] for m in range(N)])
    
    labda = np.array([labda_n(n) for n in range(N)])
    
    inv = np.linalg.inv(G)

    I1 = [sum([inv[n,j]*integrate.quad(lambda x: (h(x)-x*(1-f)-f)*phi_n(x,j), 0, 1)[0] for j in range(N)]) for n in range(N)]
    I2 = [sum([inv[n,j]*integrate.quad(lambda x: (P)*(1-f)*phi_n(x,j), 0, 1)[0] for j in range(N)]) for n in range(N)]
    
    tempi1 = np.array([np.exp(-labda[n]*tau) for n in range(N)])
    tempi2 = np.array([(1-np.exp(-labda[n]*tau))/labda[n] for n in range(N)])
    
    T = np.zeros((len(tau), N))
    
    for i in range(len(tau)):
        T1 = tempi1[:,i] * I1
        T2 = tempi2[:,i] * I2
        T[i,:] = T1 + T2
      
    def ksi(rho):
        return (1-f)*rho + f
    
    KSI = np.zeros((len(tau), len(rho)))
    for i in range(len(tau)):
        for j in range(len(rho)):
            KSI[i,j] = ksi(rho[j])
    
    sol = T @ np.array([phi_n(rho,n) for n in range(N)]) + KSI

    s0 = h(rho)
    s9 = ss(rho)

    # deltas = s9 - s0
    deltat = np.zeros(len(rho))
    # if np.sum(deltas)  < 0:
    #     for i in range(len(xi)):
    #         for j in range(len(tau)):
    #             if s9[i] - sol[j][i] >= deltas[i]*perc and deltat[i] == 0:
    #                 deltat[i] = tau[j]
    # else:
    #     for i in range(len(xi)):
    #         for j in range(len(tau)):
    #             if s9[i] - sol[j][i] <= deltas[i]*perc and deltat[i] == 0:
    #                 deltat[i] = tau[j]
  
    dstot = 0
    Stot0 = integrate.trapz(h(rho)*rho, rho)
    Stot9 = integrate.trapz(ss(rho)*rho, rho)
    for j in range(len(tau)):
        if abs(Stot9 - integrate.trapz(sol[j]*rho, rho)) < perc*abs(Stot9 - Stot0) and dstot ==0:
            dstot = tau[j]
            T_labda = -np.log(perc)/labda
            return dstot, T_labda[0], T_labda[1]
    return None

P = np.linspace(0.2,0.5,2)
q = 0.05
mplus = np.array([0.5]) # np.linspace(2,4,3)
mspace = np.append(mplus[::-1], 1/mplus)
ds, tl0, tl1 = [np.zeros(len(P)) for i in range(len(mspace))], [np.zeros(len(P)) for i in range(len(mspace))], [np.zeros(len(P)) for i in range(len(mspace))]

for j in range(len(mspace)):
    print('Calculation', j, '/', len(mspace), 'started')
    for i in range(len(P)):
        ds[j][i], tl0[j][i], tl1[j][i] = dmlts(P[i],q,mspace[j])


fig, ax = plt.subplots(1,1, figsize=(8,8))
for j in range(len(mspace)):
    ax.plot(P, ds[j], 'C'+str(j)+'.', label='m = ' + str(mspace[j]))
ax.plot(P, tl0[j], 'k-', label='ts0')
ax.plot(P, tl1[j], 'k--', label='ts1')
ax.set_xlim(0,P[-1])
ax.set_xlabel(r'Pe')
ax.set_ylabel(r'e-folding $tau$ (s)')
ax.legend()
plt.show()


## parametrisation
from scipy.optimize import curve_fit

def l(M,a,b,c):
    x,y = M
    return 1/(a*x**b*y**c + pi**2)

Np = len(P)
Nm = len(mspace)
xdata = np.zeros((2,Np*Nm))
temp=0
for i in range(Np):
    for j in range(Nm):
        xdata[0,temp] = P[i]
        xdata[1,temp] = mspace[j]
        temp+=1

ydata = []
for i in range(Np):
    for j in range(Nm):
        ydata.append(ds[j][i])

popt, pcov = curve_fit(l, xdata, np.array(ydata), p0=[0,0,0])


print('a =', popt[0], '+/-', np.sqrt(pcov[0,0]))
print('b =', popt[1], '+/-', np.sqrt(pcov[1,1]))
print('c =', popt[2], '+/-', np.sqrt(pcov[2,2]))
# print('d =', popt[3], '+/-', np.sqrt(pcov[3,3]))
# print('e =', popt[4], '+/-', np.sqrt(pcov[4,4]))
# print('f =', popt[5], '+/-', np.sqrt(pcov[5,5]))

def _l(P,q,*args):
    return l((P,q), *args)

plt.figure()
c=0
for i in range(0,Np,Np//Np):
    plt.plot(mspace, [ds[j][i] for j in range(Nm)], '.', color='C'+str(c), label='Pe = ' + str(round(P[i],3)))
    plt.plot(mspace, _l(P[i],mspace, *popt), color='C'+str(c))
    c+=1
plt.xlabel('q')
plt.ylabel('$\Lambda$')
plt.legend()
plt.show()

plt.figure()
c=0
for j in range(0,Nm,Nm//Nm):
    plt.plot(P,ds[:][j], '.', color = 'C' +str(c), label='m = ' +str(round(mspace[j],3)))
    plt.plot(P, _l(P,mspace[j], *popt), color = 'C' +str(c))
    c+=1
plt.xlabel('P')
plt.ylabel('$\Lambda$')
plt.legend()
plt.show()



# labda = np.transpose(np.array(ds1[:Nm//2]))
# q = mspace[:Nm//2]
# P = Pe
# Nq = Nm//2

# fig, axes = plt.subplots(1,3, figsize=(16,4))
# fig.suptitle(str(mspace[0]) + ' < m < ' + str(mspace[-1]))

# ax = axes[0]
# im = ax.imshow(labda[::-1,:], extent=[q[0],q[-1], P[0],P[-1]], aspect='auto')
# ax.contour(q,P,labda, 20, colors='k', inline=True)
# ax.set_xlabel('q')
# ax.set_ylabel('P')
# plt.colorbar(im, ax=ax)

# ax = axes[1]
# c=0
# for i in range(0,Np,Np//5):
#     ax.plot(q, labda[i,:], '.', color='C'+str(c), label='P = ' + str(round(P[i],3)))
#     ax.plot(q, _l(P[i],q, *popt), color='C'+str(c))
#     c+=1
# ax.set_xlabel('q')
# ax.set_ylabel('$\Lambda$')
# ax.legend()

# ax = axes[2]
# c=0
# for j in range(0,Nq, Nq//5):
#     ax.plot(P,labda[:,j], '.', color = 'C' +str(c), label='q = ' +str(round(q[j],3)))
#     ax.plot(P, _l(P,q[j], *popt), color = 'C' +str(c))
#     c+=1
# ax.set_xlabel('P')
# ax.set_ylabel('$\Lambda$')
# ax.legend()

# #plt.savefig('save_dir/figures/qlarge')

# plt.show()