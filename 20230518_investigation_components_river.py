#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:14:18 2023

@author: hugo
"""

import numpy as np
from numpy import pi
from scipy import integrate

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



Q, A, k, kappa, L, N_x = 150, 2000, 400, 400, 10000, 10


m = 400

x = np.linspace(0,L,1000)
t = np.linspace(0,1e5,m)


def phi_x_n(x,n):
    n+=1
    return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)

def labda_x_n(n):
    n+=1
    return (Q/A)**2/(4*k) + k*(n*pi/L)**2

def inner_x(n,m):
    return integrate.quad(lambda x: phi_x_n(x,m)*phi_x_n(x,n), 0, L)[0]

G_x = np.array([[inner_x(n,m) for n in range(N_x)] for m in range(N_x)])

labda_x = np.array([labda_x_n(n) for n in range(N_x)])

inv_x = np.linalg.inv(G_x)

def f(t):
    alpha = 0.5
    beta = 0.5
    return alpha + beta * np.exp(-1e-4*t)

def h(x):
    temp = np.exp(-L*0.5*Q/(A*k))
    return 1-x**2/L**2 #- phi_x_n(x,0)/10
    return f(np.inf)/(1-temp)*(np.exp(-0.5*Q*x/(k*A)) - temp)

def ssx(x):
    temp = np.exp(-L*Q/(A*k))
    return f(np.inf)/(1-temp)*(np.exp(-Q*x/(k*A)) - temp)


I1 = [sum([inv_x[n,j]*integrate.quad(lambda x: h(x)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
I2 = [sum([inv_x[n,j]*integrate.quad(lambda x: (1-x/L)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
I3 = [sum([inv_x[n,j]*integrate.quad(lambda x: (-Q/(A*L))*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]


tempix1 = np.array([I1[n] for n in range(N_x)])
tempix2 = np.array([-I2[n] for n in range(N_x)])
tempix3 = np.array([labda_x[n]*I2[n] for n in range(N_x)])
tempix4 = np.array([I3[n] for n in range(N_x)])


tempitx1 = np.array([np.exp(-labda_x[n]*t) for n in range(N_x)])
tempitx2 = np.array([[f(T) for T in t] for n in range(N_x)])
tempitx3 = np.array([[integrate.quad(lambda tau: np.exp(labda_x[n]*(tau-T)) * f(tau), 0, T)[0] for T in t] for n in range(N_x)])

Tx = np.zeros((len(t), N_x))

T1 = np.zeros((len(t), N_x))
T2 = np.zeros((len(t), N_x))
T3 = np.zeros((len(t), N_x))
T4 = np.zeros((len(t), N_x))

for i in range(len(t)):
    T1[i,:] = tempitx1[:,i] * tempix1
    T2[i,:] = tempitx2[:,i] * tempix2
    T3[i,:] = tempitx3[:,i] * tempix3
    T4[i,:] = tempitx3[:,i] * tempix4
    Tx[i,:] = T1[i,:] + T2[i,:] + T3[i,:] + T4[i,:]
    
def ksi(x,t):
    return f(t)*(1-x/L)

KSI = np.zeros((len(t), len(x)))
for i in range(len(t)):
    for j in range(len(x)):
        KSI[i,j] = ksi(x[j], t[i])
        
solx = Tx @ np.array([phi_x_n(x,n) for n in range(N_x)]) + KSI

T1sol = T1 @ np.array([phi_x_n(x,n) for n in range(N_x)]) 
T2sol = T2 @ np.array([phi_x_n(x,n) for n in range(N_x)]) 
T3sol = T3 @ np.array([phi_x_n(x,n) for n in range(N_x)]) 
T4sol = T4 @ np.array([phi_x_n(x,n) for n in range(N_x)]) 
        
s0 = h(x)
sn = ssx(x)


fig, axes = plt.subplots(2,1, figsize=(8,8))

ax = axes[0]
ax.plot(x, s0)
ax.plot(x, sn)
ax.set_xlim(0,L)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'$s$')

ax = axes[1]
ax.plot(x, KSI[0, :], label='ksi')
ax.plot(x, T1sol[0,:], label='h')
ax.plot(x, T2sol[0,:], '--', label='1-x/L')
ax.plot(x, T3sol[0,:])
ax.plot(x, T4sol[0,:])
ax.set_xlim(0,L)
ax.set_ylim(-1, 1)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$s$')
ax.legend()
        
# Plot the surface.
def init():
    ax = axes[0]
    ax.plot(x, s0, label='initial')
    ax.plot(x, sn, label='final')
    ax.set_xlim(0,L)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    
    ax = axes[1]
    ax.plot(x, KSI[0, :], label='ksi')
    ax.plot(x, T1sol[0,:], label='h')
    ax.plot(x, T2sol[0,:], '--', label='1-x/L')
    ax.plot(x, T3sol[0,:])
    ax.plot(x, T4sol[0,:])
    ax.set_xlim(0,L)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.legend()
    return 

# animation function.  This is called sequentially
def animate(i):
    ax = axes[0]
    ax.clear()
    ax.plot(x, s0, label='initial')
    ax.plot(x, sn, label='final')
    ax.plot(x, solx[i], label=str(t[i]))
    ax.set_xlim(0,L)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.legend()
    
    ax = axes[1]
    ax.clear()
    ax.plot(x, KSI[i, :], label='ksi')
    ax.plot(x, T1sol[i,:], label='h')
    ax.plot(x, T2sol[i,:], label='1-x/L')
    ax.plot(x, T3sol[i,:], label='time 1-x/L')
    ax.plot(x, T4sol[i,:], label='deviation')
    ax.set_xlim(0,L)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=m, interval=10, repeat=False)
plt.show()


barx = np.array([n for n in range(N_x)])

H = np.array([integrate.quad(lambda x: h(x)*phi_x_n(x, n), 0, L)[0] for n in range(N_x)])
H2 = np.array([integrate.quad(lambda x: (1-x/L)*phi_x_n(x, n), 0, L)[0] for n in range(N_x)])

plt.figure()
plt.bar(barx,H)
plt.bar(barx,H2, width=0.4)
plt.show()

xs = np.linspace(0,L,1000)
plt.figure()
for n in range(10):
    plt.plot(xs, phi_x_n(xs,n))
plt.show()    