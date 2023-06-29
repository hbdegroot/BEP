#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:59:42 2023

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
t = np.linspace(0,1e6,1000 + 1)

def ss(x):
    temp = np.exp(-L*Q/(A*k))
    return f/(1-temp)*(np.exp(-Q*x/(k*A)) - temp)

def h1(x):
    temp = np.exp(-L*0.5*Q/(A*k))
    return f/(1-temp)*(np.exp(-0.5*Q*x/(k*A)) - temp)

def h2(x):
    temp = np.exp(-L*2*Q/(A*k))
    #return ss(x) - np.sin(pi*x/L)/10
    return f/(1-temp)*(np.exp(-2*Q*x/(k*A)) - temp)

def phi_n(x,n):
    n+=1
    return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)


def labda_n(n):
    n+=1
    return (Q/A)**2/(4*k) + k*(n*pi/L)**2

def inner(n,m):
    return integrate.quad(lambda x: phi_n(x,m)*phi_n(x,n), 0, L)[0]

G = np.array([[inner(n,m) for n in range(N)] for m in range(N)])

labda = np.array([labda_n(n) for n in range(N)])

inv = np.linalg.inv(G)

def solf(h):
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

    s0 = h(x)
    s9 = ss(x)

    deltas = s9 - s0
    deltat = np.zeros(len(x))
    if np.sum(deltas)  < 0:
        for i in range(len(x)):
            for j in range(len(t)):
                if s9[i] - sol[j][i] >= deltas[i]*perc and deltat[i] == 0:
                    deltat[i] = t[j]
    else:
        for i in range(len(x)):
            for j in range(len(t)):
                if s9[i] - sol[j][i] <= deltas[i]*perc and deltat[i] == 0:
                    deltat[i] = t[j]
  
    dstot = 0
    Stot0 = integrate.quad(h, 0, L)[0]
    Stot9 = integrate.quad(ss, 0, L)[0]
    for j in range(len(t)):
        if abs(Stot9 - integrate.trapz(sol[j], x)) < perc*abs(Stot9 - Stot0) and dstot ==0:
            dstot = t[j]
            
    return sol, deltat, T, dstot


sol1, deltat1, T1, dstot1 = solf(h1)
sol2, deltat2, T2, dstot2 = solf(h2)

T_labda = -np.log(perc)/labda

 


fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(-x, deltat1, 'C0.', markersize='4', label=r'$t_{ADJ}$ (decreasing $s$)')
ax.plot(-x, deltat2, 'C1.', markersize='4', label=r'$t_{ADJ}$ (increasing $s$)')
ax.hlines(T_labda[0], -L, 0, colors='k', label = r'$1/ \lambda_1$')
#ax.hlines(L*A/Q, 0, L, colors='r', label='$t_p$')
#ax.hlines(0.5*L*A/Q, 0, L, colors='r', label='0.5$t_p$')
ax.hlines(dstot1, -L, 0, colors='C0', label=r'$T_{ADJ}$ (decreasing $S$)')
ax.hlines(dstot2, -L, 0, colors='C1', label=r'$T_{ADJ}$ (increasing $S$)')
ax.set_xlim(-L,0)
ax.set_ylim(0,275000)
ax.set_xlabel(r'$-x$ (m)')
ax.set_ylabel(r'$t$ (s)')
ax.legend()
plt.show()


fig, axes = plt.subplots(1,2, figsize=(12,8))
ax = axes[0]
for j in range(10):
    ax.plot(t, T1[:,j], 'C'+str(j), label='n='+str(int(j+1)))
ax.set_xlim(0,1e6)
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$T_n(t)$')
#ax.set_yscale('log')
ax.legend()

ax = axes[1]
for j in range(10):
    ax.plot(t, T1[:,j]-T1[-1,j])
ax.set_xlim(0,1e6)
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$T_n(t)-T_n(\infty)$')
#ax.set_yscale('log')

plt.show()

fig, axes = plt.subplots(1,2, figsize=(12,8))
ax = axes[0]
for j in range(10):
    ax.plot(t, T2[:,j], 'C'+str(j), label='n='+str(int(j+1)))
ax.set_xlim(0,1e6)
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$T_n(t)$')
#ax.set_yscale('log')
ax.legend()

ax = axes[1]
for j in range(10):
    ax.plot(t, T2[:,j]-T2[-1,j])
ax.set_xlim(0,1e6)
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$T_n(t)-T_n(\infty)$')
#ax.set_yscale('log')
#ax.legend()

plt.show()


fig, axes = plt.subplots(1,2, figsize=(12,8))
ax = axes[0]
for j in range(10):
    ax.plot(t, T1[:,j], 'C'+str(j), label='n='+str(int(j+1)))
ax.set_xlim(0,1e6)
ax.set_ylim(-1, 0.1)
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$T_n(t)$')
ax.set_title('decreasing $s$')
#ax.set_yscale('log')
ax.legend()
ax = axes[1]
for j in range(10):
    ax.plot(t, T2[:,j], 'C'+str(j), label='n='+str(int(j+1)))
ax.set_xlim(0,1e6)
ax.set_ylim(-1, 0.1)
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$T_n(t)$')
ax.set_title('increasing $s$')
#ax.set_yscale('log')
ax.legend()
plt.show()




s01 = h1(x)
s02 = h2(x)
s9 = ss(x)

fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(x, s01, 'C0--')
ax.plot(x, s02, 'C1--')
ax.plot(x, s9, 'k-')
ax.set_xlim(0,L)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$s$')

# Plot the surface.
def init():
    ax.plot(x, s01, label='initial1')
    ax.plot(x, s02, label='initial2')
    ax.plot(x, s9, label='final')
    ax.set_xlim(0,L)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax.clear()
    ax.plot(x, s01, 'C0--', label='initial1')
    ax.plot(x, s02, 'C1--', label='initial2')
    ax.plot(x, s9 - (s9-s01)/np.e, 'k:')
    ax.plot(x, s9 - (s9-s02)/np.e, 'k:')
    ax.plot(x, s9, 'k-', label='final')
    ax.plot(x, sol1[i], 'C0-')
    ax.plot(x, sol2[i], 'C1-')
    ax.set_xlim(0,L)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.set_title(str(t[i]))
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=len(t), interval=10, repeat=False)
plt.show()

x_ = -x
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(x_, s01, 'k--', label=r'$s(t=0)$')
ax.plot(x_, s9 - (s9-s01)/np.e, 'k:', label = r'$s(t=t_{ADJ})$')
ax.plot(x_, s9, 'k-', label=r'$s(t = \infty)$')
ax.plot(x_, sol1[len(t)//20], 'C0-', label = '$s(t$ = ' + str(int(t[len(t)//20])) + ' s)')
ax.plot(x_, sol1[len(t)//8], 'C1-', label = '$s(t$ = ' + str(int(t[len(t)//8])) + ' s)')
ax.plot(x_, sol1[len(t)//4], 'C2-', label = '$s(t$ = ' + str(int(t[len(t)//4])) + ' s)')
ax.set_xlim(-L,0)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$-x$ (m)')
ax.set_ylabel(r'$s/s_{sea}$')
ax.legend()
plt.show()