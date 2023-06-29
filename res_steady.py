#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:10:33 2023

@author: hugo
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

Q = 500
A = 7500
k = 900
kappa = 180
L = 45000
R = 7000
D = 20


a = A/(np.pi*D)
x = np.linspace(0,L,500)
r = np.linspace(a,R,500)
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


def ss(x,r, Q, A, k, kappa, R, L, D):
    return np.append(ssx(x, Q, A, k, kappa, R, L, D)[::-1], ssr(r, Q, A, k, kappa, R, L, D))

varspace1 = np.array([kappa/2, kappa, kappa*2])
varspace2 = np.array([R/2, R, R*2])
varspace3 = np.array([D/2, D, D*2])


# fig, axes = plt.subplots(3,1, figsize=(8,10))

# ax = axes[0]
# for i in range(len(varspace1)):
#     kappavar = varspace1[i]
    
#     a = A/(np.pi*D)
#     x = np.linspace(0,L,500)
#     r = np.linspace(a,R,500)
#     axs = np.append(-x[::-1], r)

#     ax.plot(axs, ss(x,r,Q, A, k, kappavar, R, L, D), '-', color='C' + str(i), label='$\kappa$ = '+str(int(varspace1[i])) + ' m$^2$ s$^{-1}$')

# ax.set_xlim(-L, R)
# ax.set_ylim(0,1)
# ax.set_xlabel(r'$-x & r$ (m)')
# ax.set_ylabel(r'$s/s_{sea}$')
# ax.legend()

# ax = axes[1]
# for i in range(len(varspace2)):
#     Rvar = varspace2[i]
    
#     a = A/(np.pi*D)
#     x = np.linspace(0,L,500)
#     r = np.linspace(a,R,500)
#     axs = np.append(-x[::-1], r)
    
#     ax.plot(axs, ss(x,r,Q, A, k, kappa, Rvar, L, D), '-', color='C' + str(i), label='$R$ = '+str(int(varspace2[i])) + ' m')

# ax.set_xlim(-L, R)
# ax.set_ylim(0,1)
# ax.set_xlabel(r'$-x & r$ (m)')
# ax.set_ylabel(r'$s/s_{sea}$')
# ax.legend()

# ax = axes[2]
# for i in range(len(varspace3)):
#     Dvar = varspace3[i]
    
#     a = A/(np.pi*Dvar)
#     x = np.linspace(0,L,500)
#     r = np.linspace(a,R,500)
#     axs = np.append(-x[::-1], r)

#     ax.plot(axs, ss(x,r,Q, A, k, kappa, R, L, Dvar), '-', color='C' + str(i), label='$D$ = '+str(int(varspace3[i])) + ' m')

# ax.set_xlim(-L, R)
# ax.set_ylim(0,1)
# ax.set_xlabel(r'$-x & r$ (m)')
# ax.set_ylabel(r'$s/s_{sea}$')
# ax.legend()

# plt.show()



kapspace = np.linspace(50,4000,500)
Dspace = np.linspace(1,50,500)
Rspace = np.linspace(100, 50000,500)

# fig, axes = plt.subplots(1,3, figsize=(8,4))
# ax = axes[0]
# ax.plot(kapspace, ssx(0,Q, A, k, kapspace, R, L, D), '-', color='C0')
# ax.set_ylim(0,1)
# ax.set_xlabel(r'$-x & r$ (m)')
# ax.set_ylabel(r'$s(0)/s_{sea}$')
# ax.legend()

# ax = axes[1]
# ax.plot(Rspace, ssx(0,Q, A, k, kappa, Rspace, L, D), '-', color='C1')
# ax.set_ylim(0,1)
# ax.set_xlabel(r'$-x & r$ (m)')
# ax.set_ylabel(r'$s(0)/s_{sea}$')
# ax.legend()

# ax = axes[2]
# ax.plot(Dspace, ssx(0,Q, A, k, kappa, R, L, Dspace), '-', color='C0')
# ax.set_ylim(0,1)
# ax.set_xlabel(r'$-x & r$ (m)')
# ax.set_ylabel(r'$s(0)/s_{sea}$')
# ax.legend()

# plt.show()



fig, axes = plt.subplots(3,2, figsize=(12,12), gridspec_kw={'width_ratios': [2, 1]})

ax = axes[0,0]
for i in range(len(varspace1)):
    kappavar = varspace1[i]
    
    a = A/(np.pi*D)
    x = np.linspace(0,L,500)
    r = np.linspace(a,R,500)
    axs = np.append(-x[::-1], r)

    ax.plot(axs, ss(x,r,Q, A, k, kappavar, R, L, D), '-', color='C' + str(i), label='$\kappa$ = '+str(int(varspace1[i])) + ' m$^2$ s$^{-1}$')

ax.set_xlim(-L, R)
ax.set_ylim(0,1)
ax.set_xlabel(r'$-x & r$ (m)')
ax.set_ylabel(r'$s/s_{sea}$')
ax.legend()

ax = axes[0,1]
ax.plot(kapspace, ssx(0,Q, A, k, kapspace, R, L, D), '-', color='C0')
ax.set_xlim(0,kapspace[-1])
ax.set_ylim(0,1)
ax.set_xlabel(r'$\kappa$ (m$^2$ s$^{-1}$)')
ax.set_ylabel(r'$s(0)/s_{sea}$')

ax = axes[1,0]
for i in range(len(varspace2)):
    Rvar = varspace2[i]
    
    a = A/(np.pi*D)
    x = np.linspace(0,L,500)
    r = np.linspace(a,R,500)
    axs = np.append(-x[::-1], r)
    
    ax.plot(axs, ss(x,r,Q, A, k, kappa, Rvar, L, D), '-', color='C' + str(i), label='$R$ = '+str(int(varspace2[i])) + ' m')

ax.set_xlim(-L, R)
ax.set_ylim(0,1)
ax.set_xlabel(r'$-x & r$ (m)')
ax.set_ylabel(r'$s/s_{sea}$')
ax.legend()

ax = axes[1,1]
ax.plot(Rspace, ssx(0,Q, A, k, kappa, Rspace, L, D), '-', color='C0')
ax.set_xlim(0,Rspace[-1])
ax.set_ylim(0,1)
ax.set_xlabel(r'$R$ (m)')
ax.set_ylabel(r'$s(0)/s_{sea}$')

ax = axes[2,0]
for i in range(len(varspace3)):
    Dvar = varspace3[i]
    
    a = A/(np.pi*Dvar)
    x = np.linspace(0,L,500)
    r = np.linspace(a,R,500)
    axs = np.append(-x[::-1], r)

    ax.plot(axs, ss(x,r,Q, A, k, kappa, R, L, Dvar), '-', color='C' + str(i), label='$D$ = '+str(int(varspace3[i])) + ' m')

ax.set_xlim(-L, R)
ax.set_ylim(0,1)
ax.set_xlabel(r'$-x & r$ (m)')
ax.set_ylabel(r'$s/s_{sea}$')
ax.legend()

ax = axes[2,1]
ax.plot(Dspace, ssx(0,Q, A, k, kappa, R, L, Dspace), '-', color='C0')
ax.set_xlim(0,Dspace[-1])
ax.set_ylim(0,1)
ax.set_xlabel(r'$D$ (m)')
ax.set_ylabel(r'$s(0)/s_{sea}$')

plt.show()
