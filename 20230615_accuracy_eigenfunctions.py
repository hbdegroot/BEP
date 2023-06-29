#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:01:34 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import integrate

pi = np.pi


Q, A, k, L, N = 500, 7500, 900, 45000, 50

f = 0.8
perc = 1/np.e

x = np.linspace(0,L,1000)
t = np.linspace(0,1e7,10 + 1)

def ss(x):
    temp = np.exp(-L*Q/(A*k))
    return f/(1-temp)*(np.exp(-Q*x/(k*A)) - temp)

def h(x):
    temp = np.exp(-L*0.5*Q/(A*k))
    return f/(1-temp)*(np.exp(-0.5*Q*x/(k*A)) - temp)

def phi_n(x,n):
    n+=1
    return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)


def labda_n(n):
    n+=1
    return (Q/A)**2/(4*k) + k*(n*pi/L)**2

def inner(n,m):
    return integrate.quad(lambda x: phi_n(x,m)*phi_n(x,n), 0, L)[0]

def solver(N):
    G = np.array([[inner(n,m) for n in range(N)] for m in range(N)]) 
    labda = np.array([labda_n(n) for n in range(N)])
    inv = np.linalg.inv(G)
    I1 = [sum([inv[n,j]*integrate.quad(lambda x: (h(x)-f*(1-x/L))*phi_n(x,j), 0, L)[0] for j in range(N)]) for n in range(N)]
    I2 = [sum([inv[n,j]*integrate.quad(lambda x: (-Q/A*f/L)*phi_n(x,j), 0, L)[0] for j in range(N)]) for n in range(N)]
    
    tempi1 = np.array([np.exp(-labda[n]*t) for n in range(N)])
    tempi2 = np.array([(1-np.exp(-labda[n]*t))/labda[n] for n in range(N)])
    
    T = np.zeros((len(t), N))
    
    for i in range(len(t)):
        T1 = tempi1[:,i] * I1
        T2 = tempi2[:,i] * I2
        T[i,:] = T1 + T2
      
    def ksi(x):
        return f*(1-x/L)
    
    KSI = np.zeros((len(t), len(x)))
    for i in range(len(t)):
        for j in range(len(x)):
            KSI[i,j] = ksi(x[j])
    
    sol = T @ np.array([phi_n(x,n) for n in range(N)]) + KSI          
    return sol[0], sol[-1]


sol11, sol12 = solver(10)
sol21, sol22 = solver(20)
sol31, sol32 = solver(30)
sol41, sol42 = solver(50)

s0 = h(x)
sn = ss(x)

fig, axes = plt.subplots(1,4, figsize=(8,16))
ax = axes[0]
ax.plot(x, s0-sol11, color='C1', ls='-', label=r'$t=0$ s')
ax.plot(x, sn-sol12, color='C2', ls='--', label=r'$t = 10^7$ s')
ax.set_xlabel(r'$x$ (m)')
ax.set_ylabel(r'$\Delta \zeta$')
ax.set_title('N = 10')
ax.legend()

ax = axes[1]
ax.plot(x, s0-sol21, color='C1', ls='-', label=r'$t=0$ s')
ax.plot(x, sn-sol22, color='C2', ls='--', label=r'$t = 10^7$ s')
ax.set_xlabel(r'$x$ (m)')
ax.set_ylabel(r'$\Delta \zeta$')
ax.set_title('N = 20')

ax = axes[2]
ax.plot(x, s0-sol31, color='C1', ls='-', label=r'$t=0$ s')
ax.plot(x, sn-sol32, color='C2', ls='--', label=r'$t = 10^7$ s')
ax.set_xlabel(r'$x$ (m)')
ax.set_ylabel(r'$\Delta \zeta$')
ax.set_title('N = 30')

ax = axes[3]
ax.plot(x, s0-sol41, color='C1', ls='-', label=r'$t=0$ s')
ax.plot(x, sn-sol42, color='C2', ls='--', label=r'$t = 10^7$ s')
ax.set_xlabel(r'$x$ (m)')
ax.set_ylabel(r'$\Delta \zeta$')
ax.set_title('N = 50')
plt.show()


import matplotlib.patches as mpatches
import matplotlib.lines as mlines

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(x, sn-sol11, color='C0', ls='-')
ax.plot(x, sn-sol12, color='C0', ls='--')

ax.plot(x, sn-sol21, color='C1', ls='-')
ax.plot(x, sn-sol22, color='C1', ls='--')

ax.plot(x, sn-sol31, color='C2', ls='-')
ax.plot(x, sn-sol32, color='C2', ls='--')

ax.plot(x, sn-sol41, color='C3', ls='-')
ax.plot(x, sn-sol42, color='C3', ls='--')

ax.set_xlabel(r'$-x$ (m)')
ax.set_ylabel(r'$[s(\infty) - s(10^7$s$)]/s_{sea}$')
#ax.set_title(r'$\Delta x = $' + str(dx3) + 'm, $\Delta r =$ ' + str(dr3))

dt_line = mlines.Line2D([], [], color='k', ls='-', label=r't=0')
dt2_line = mlines.Line2D([], [], color='k', ls='--', label=r't=10^6')

c0_patch = mpatches.Patch(color='C0', label=r'N=10')
c1_patch = mpatches.Patch(color='C1', label=r'N=20')
c2_patch = mpatches.Patch(color='C2', label=r'N=20')
plt.legend(handles=[dt_line, dt2_line, c0_patch, c1_patch, c2_patch])

#ax.legend()
plt.show()
