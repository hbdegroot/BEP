#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 11:02:12 2023

@author: hugo
"""

import numpy as np
from numpy import pi
from scipy import integrate

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



Q, A, k, kappa, L, N_x = 150, 2000, 400, 400, 10000, 50


m = 400

x = np.linspace(0,L,500)
t = np.linspace(0,1e5,m)


def phi_x_n(x,n):
    n+=1
    return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)

def dphi_dx_n(x,n):
    n+=1
    return n*pi/L

def labda_x_n(n):
    n+=1
    return (Q/A)**2/(4*k) + k*(n*pi/L)**2

def inner_x(n,m):
    return integrate.quad(lambda x: phi_x_n(x,m)*phi_x_n(x,n), 0, L)[0]

G_x = np.array([[inner_x(n,m) for n in range(N_x)] for m in range(N_x)])

labda_x = np.array([labda_x_n(n) for n in range(N_x)])

inv_x = np.linalg.inv(G_x)

tc = 1e-4
def f(t):
    alpha = 1
    beta = -0.5
    return alpha + beta * np.exp(-tc*t)

def h(x):
    temp = np.exp(-L*0.5*Q/(A*k))
    return 1-x/L #- phi_x_n(x,0)/10
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


numdsdx0 = np.zeros(len(t))
for j in range(len(t)):
    numdsdx0[j] = (solx[j][1] - solx[j][0])/(x[1]-x[0])

def ksi_x(t):
    return -f(t)/L

KSI_x = np.zeros(len(t))
for i in range(len(t)):
    KSI_x[i] = ksi_x(t[i])
        
dsdx0 = Tx @ np.array([dphi_dx_n(0,n) for n in range(N_x)]) + KSI_x


fig, axes = plt.subplots(1,1, figsize=(8,8))
ax = axes
ax.plot(t, numdsdx0)
ax.plot(t, dsdx0)
ax.set_xlim(0,t[-1])
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$s_x(0)$')
plt.show()


par = np.polyfit(t, dsdx0, 25)

g = np.poly1d(par)

plt.figure()
plt.plot(t, g(t))
plt.show()

from scipy.optimize import root
def trapz(f_arr, f_new, t_arr, t_new, n):
    if len(f_arr)==1:
        return dt/2*(f_arr[0]*np.exp(-labda_x[n]*t_new + f_new))
        
    s = 0
    for i in range(1, len(f_arr)):
        s += f_arr[i] * np.exp(labda_x[n] * (t_arr[i]-t_new))
    return dt/2*(f_arr[0]*np.exp(-labda_x[n]*t_new + f_new + 2*s))
                 
                 
dt = 0
f_arr = [f(0)]
t_arr = [0]

N_steps = 100

for i in range(1,N_steps):
    t_new = max(t_arr) + dt
    def rootf(f_new):
        return -f_new/L + sum([n*pi/L * (np.exp(-labda_x[n]*t_new) * I1[n] - f_new*I2[n] + trapz(f_arr, f_new, t_arr, t_new, n) * (tempix3[n] + I3[n])) for n in range(N_x)]) + g(t_new)
    f_new = root(rootf, 0.9*f(t_new)).x
    
    f_arr.append(f_new)
    t_arr.append(t_new)

plt.figure()
plt.plot(t_arr, f_arr)
plt.plot(t_arr, [f(t_arr[i]) for i in range(len(t_arr))])
plt.xlabel('t')
plt.ylabel('approx f(t)')
plt.show()


dt = 0
f_arr = [f(0)]
t_arr = [0]

def inves(f_new):
    return -f_new/L + sum([n*pi/L * (np.exp(-labda_x[n]*t_new) * I1[n] - f_new*I2[n] + trapz(f_arr, f_new, t_arr, t_new, n) * (tempix3[n] + I3[n])) for n in range(N_x)])

fnew  = np.linspace(0,1, 50)
plt.figure()
plt.plot(fnew, [inves(f) for f in fnew])
plt.plot(fnew, [g(dt) for f in fnew])
plt.xlabel('f_new')
plt.show()