#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:38:13 2023

@author: hugo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import integrate
from scipy.special import jv
from scipy.special import yv
from module_sol_v1 import goodrootfinder

pi = np.pi

Q, D, A, kappa, R, N = 2000, 20, 7500, 100, 7000, 50

f = 0.8
perc = 1/np.e

a = A/(pi*D)

r = np.linspace(a,R,1000)
t = np.linspace(0,1e7,10+1)

def ss(r):
    o = Q/(kappa*pi*D)
    temp = (1-f)/(R**o - a**o)
    return temp*(r**o-R**o) + 1

def h(r):
    o = 0.5*Q/(kappa*pi*D)
    temp = (1-f)/(R**o - a**o)
    return temp*(r**o-R**o) + 1 #- np.sin((r-a)/(R-a)*pi)/2

print('Looking for some roots...')
good_roots = goodrootfinder(N, 1e-5, 0.1, 1000, Q, A, kappa, R, 0, D)
print('All set, continue integrating \n')

order = Q/(2*kappa*pi*D)
def phi_r(r,par):
    return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))

sign = [-np.sign(phi_r(0.01, good_roots[n]))/np.sqrt(integrate.quad(lambda r: phi_r(r,good_roots[n])**2, a, R)[0]) for n in range(N)]

def phi_n(r,n):
    return phi_r(r, good_roots[n]) * sign[n]

def labda_n(n):
    return good_roots[n][0]

def inner(n,m):
    return integrate.quad(lambda r: phi_n(r,m)*phi_n(r,n), a, R)[0]

def solver(N):
    G = np.array([[inner(n,m) for n in range(N)] for m in range(N)])
    labda = np.array([labda_n(n) for n in range(N)])
    inv = np.linalg.inv(G)

    I1 = [sum([inv[n,j]*integrate.quad(lambda rho: (h(rho)-(1-f)/(R-a)*rho - (R*f-a)/(R-a))*phi_n(rho,j), a, R)[0] for j in range(N)]) for n in range(N)]
    I2 = [sum([inv[n,j]*integrate.quad(lambda r: 1/r*(kappa-Q/(pi*D))*(1-f)/(R-a)*phi_n(r,j), a, R)[0] for j in range(N)]) for n in range(N)]
    
    tempi1 = np.array([np.exp(-labda[n]*t) for n in range(N)])
    tempi2 = np.array([(1-np.exp(-labda[n]*t))/labda[n] for n in range(N)])
    
    T = np.zeros((len(t), N))
    
    for i in range(len(t)):
        T1 = tempi1[:,i] * I1
        T2 = tempi2[:,i] * I2
        T[i,:] = T1 + T2
      
    def ksi(r):
        return (1-f)/(R-a)*r + (R*f-a)/(R-a)
    
    KSI = np.zeros((len(t), len(r)))
    for i in range(len(t)):
        for j in range(len(r)):
            KSI[i,j] = ksi(r[j])
    
    sol = T @ np.array([phi_n(r,n) for n in range(N)]) + KSI
                
    return sol[0], sol[-1]


sol11, sol12 = solver(10)
sol21, sol22 = solver(20)
sol31, sol32 = solver(30)
sol41, sol42 = solver(50)

s0 = h(r)
sn = ss(r)

fig, axes = plt.subplots(2,2, figsize=(15,10))
ax = axes[0,0]
ax.plot(r, s0-sol11, color='C1', ls='-', label=r'$t=0$ s')
ax.plot(r, sn-sol12, color='C2', ls='--', label=r'$t = 10^7$ s')
ax.set_xlabel(r'$r$ (m)')
ax.set_ylabel(r'$\Delta \zeta$')
ax.set_xlim(a,R)
ax.set_title('N = 10')
ax.legend()

ax = axes[0,1]
ax.plot(r, s0-sol21, color='C1', ls='-', label=r'$t=0$ s')
ax.plot(r, sn-sol22, color='C2', ls='--', label=r'$t = 10^7$ s')
ax.set_xlabel(r'$r$ (m)')
ax.set_ylabel(r'$\Delta \zeta$')
ax.set_xlim(a,R)
ax.set_title('N = 20')

ax = axes[1,0]
ax.plot(r, s0-sol31, color='C1', ls='-', label=r'$t=0$ s')
ax.plot(r, sn-sol32, color='C2', ls='--', label=r'$t = 10^7$ s')
ax.set_xlabel(r'$r$ (m)')
ax.set_ylabel(r'$\Delta \zeta$')
ax.set_xlim(a,R)
ax.set_title('N = 30')

ax = axes[1,1]
ax.plot(r, s0-sol41, color='C1', ls='-', label=r'$t=0$ s')
ax.plot(r, sn-sol42, color='C2', ls='--', label=r'$t = 10^7$ s')
ax.set_xlabel(r'$r$ (m)')
ax.set_ylabel(r'$\Delta \zeta$')
ax.set_xlim(a,R)
ax.set_title('N = 50')
plt.show()


# import matplotlib.patches as mpatches
# import matplotlib.lines as mlines

# fig, ax = plt.subplots(1,1, figsize=(8,8))
# ax.plot(x, sn-sol11, color='C0', ls='-')
# ax.plot(x, sn-sol12, color='C0', ls='--')

# ax.plot(x, sn-sol21, color='C1', ls='-')
# ax.plot(x, sn-sol22, color='C1', ls='--')

# ax.plot(x, sn-sol31, color='C2', ls='-')
# ax.plot(x, sn-sol32, color='C2', ls='--')

# ax.plot(x, sn-sol41, color='C3', ls='-')
# ax.plot(x, sn-sol42, color='C3', ls='--')

# ax.set_xlabel(r'$-x$ (m)')
# ax.set_ylabel(r'$[s(\infty) - s(10^7$s$)]/s_{sea}$')
# #ax.set_title(r'$\Delta x = $' + str(dx3) + 'm, $\Delta r =$ ' + str(dr3))

# dt_line = mlines.Line2D([], [], color='k', ls='-', label=r't=0')
# dt2_line = mlines.Line2D([], [], color='k', ls='--', label=r't=10^6')

# c0_patch = mpatches.Patch(color='C0', label=r'N=10')
# c1_patch = mpatches.Patch(color='C1', label=r'N=20')
# c2_patch = mpatches.Patch(color='C2', label=r'N=20')
# plt.legend(handles=[dt_line, dt2_line, c0_patch, c1_patch, c2_patch])

# #ax.legend()
# plt.show()
