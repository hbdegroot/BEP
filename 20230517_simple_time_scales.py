#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:35:29 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import integrate

pi = np.pi


Q, A, k, L, N = 150, 2000, 400, 20000, 5

f = 1

x = np.linspace(0,L,100)
t = np.linspace(0,1e6,1000)

def ss(x):
    temp = np.exp(-L*Q/(A*k))
    return f/(1-temp)*(np.exp(-Q*x/(k*A)) - temp)

def h1(x):
    temp = np.exp(-L*0.5*Q/(A*k))
    #return ss(x) + np.sin(pi*x/L)/10
    return f/(1-temp)*(np.exp(-0.5*Q*x/(k*A)) - temp)

def h2(x):
    temp = np.exp(-L*2*Q/(A*k))
    #return ss(x) - np.sin(pi*x/L)/10
    return f/(1-temp)*(np.exp(-2*Q*x/(k*A)) - temp)

def phi_n(x,n):
    n+=1
    return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)

# def h(x):
#     return -phi_n(x,0) + f*(1-x/L)


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
                if s9[i] - sol[j][i] >= deltas[i]/np.e and deltat[i] == 0:
                    deltat[i] = t[j]
    else:
        for i in range(len(x)):
            for j in range(len(t)):
                if s9[i] - sol[j][i] <= deltas[i]/np.e and deltat[i] == 0:
                    deltat[i] = t[j]
    return sol, deltat, T

sol1, deltat1, T1 = solf(h1)
sol2, deltat2, T2 = solf(h2)

T_labda = 1/labda

deltaT = np.zeros(len(x))
for n in range(N):
    s0 = T1[0,n]
    s9 = T1[-1,n]    
    deltas = s9 - s0
    if deltas  < 0:
        for j in range(len(t)):
            if s9 - T1[j,n] >= deltas/np.e and deltaT[n] == 0:
                deltaT[n] = t[j]
    else:
        for j in range(len(t)):
            if s9 - T1[j,n] <= deltas/np.e and deltaT[n] == 0:
                deltaT[n] = t[j]
 
    
fig, ax = plt.subplots(1,1, figsize=(8,8))
for n  in range(N):
    ax.hlines(deltaT[n] , 0, L, label=str(n))
ax.hlines(T_labda, 0, L, 'k', linestyles='dashed')
ax.set_xlim(0,L)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'T_n (t)')
ax.legend()
plt.show()               
 
fig, ax = plt.subplots(1,1, figsize=(8,8))
for n in range(N):
    ax.plot(t, T1[:,n], 'C0-')
    ax.plot(t, T2[:,n], 'C1-')
ax.plot(t, np.exp(-labda[0]*t), 'k-')
ax.set_xlim(0,t[-1])
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'T_n (t)')
ax.legend()
plt.show()


fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(x, deltat1, 'C0.', label='result')
ax.plot(x, deltat2, 'C1.', label='result')
ax.hlines(T_labda, 0, L, colors='k', label = r'$1/ \lambda$')
ax.hlines(L*A/Q, 0, L, colors='r', label='$t_p$')
ax.set_xlim(0,L)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'e-folding $t$')
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

s01 = h1(x)
s02 = h2(x)
s9 = ss(x)
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(x, s01-s9, 'C0-')
ax.plot(x, s02-s9, 'C1-')
ax.set_xlim(0,L)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$s$')

# Plot the surface.
def init():
    ax.plot(x, s01-s9, 'C0-')
    ax.plot(x, s02-s9, 'C1-')
    ax.set_xlim(0,L)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax.clear()
    ax.plot(x, sol1[i]-s9, 'C0-')
    ax.plot(x, sol2[i]-s9, 'C1-')
    ax.set_xlim(0,L)
    #ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.set_title(str(t[i]))
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=len(t), interval=10, repeat=False)
plt.show()