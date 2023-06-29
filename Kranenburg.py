#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:57:11 2023

@author: hugo
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

# Numerical Parameters
n = 100
m = 100000


k = 0.48
nprime = 1.3
labda = 84.4e-6
A0 = 2000
Q0 = 150

L = 1/labda
D = (k/labda)**(nprime+1)/A0 * Q0


dt = 10
dx = L/n
x = np.linspace(0,L,n+1)[1:-1]
t = np.arange(0,m*dt, dt)


def A(x):
    return A0 * np.exp(-labda*x)

def Q(t):
    T = min(t, m*dt/2)
    return Q0*(1+T/(m*dt))


def dsdt(si, ti):
    dsdx = M @ si + b
    ds2dx2 = K @ si + a
    return Q(ti)*dsdx/A(x) + D*ds2dx2 #- D*labda*dsdx

def phi(x,t):
    i = 

def fss(x,t):
    c1 = Q(t)/(A0*D)/(1-np.exp(-Q(t)*L/(A0*D)))
    c2 = -c1*A0*D/Q(t)*np.exp(-Q(t)*L/(A0*D))
    return c1*A0*D/Q(t)*np.exp(-Q(t)*x/(A0*D)) + c2



s = [np.zeros(n-1)]*m
ss = [np.zeros(n-1)]*m
s[0] = fss(x,0)
ss[0] = fss(x,0)

M = 1/(2*dx) * (np.diag(-1*np.ones(n-2), -1) + np.diag(np.ones(n-2), 1))
K = 1/(dx)**2 * (np.diag(-2 * np.ones(n-1), 0) + np.diag(1*np.ones(n-2), -1) + np.diag(1 * np.ones(n-2), 1))
b = np.zeros(n-1)
b[0] = -1/(2*dx)
a = np.zeros(n-1)
a[0] = 1/(dx)**2

for i in range(0,m-1):
    pred = s[i] + dt*dsdt(s[i], t[i])
    s[i+1] = s[i] + dt/2 *(dsdt(s[i], t[i]) + dsdt(pred, t[i+1]))
    ss[i+1] = fss(x,t[i+1])



# plt.figure()
# # for i in range(0,n-10, n//10):
# #     plt.plot(t, S[:,i])
# plt.plot(x, s[0], label ='t=0')
# plt.plot(x, s[1], label ='t=1')
# plt.plot(x, s[4], label ='t=4')
# plt.plot(x, s[8], label ='t=8')
# plt.plot(x, s[m//2], label = 't=T/2')
# plt.plot(x, s[-1], label = 'T')
# plt.xlabel('s(t)')
# plt.ylabel('x')
# plt.ylim(-0.1, 1.1)
# plt.legend()
# plt.show()

skips = 1000
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(x, s[0])
ax.plot(x, ss[0], label = '$ss$')
ax.set_xlim(0, L)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$s$')
ax.legend()

# Plot the surface.
def init():
    ax.clear()
    ax.plot(x, s[0], label = '$s$')
    ax.plot(x, ss[0], label = '$ss$')
    ax.set_xlim(0, L)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.legend()
    return 

# animation function.  This is called sequentially
def animate(i):
    ax.clear()
    ax.plot(x, s[i*skips], 'C0-', label = '$s$')
    ax.plot(x, ss[i*skips], 'C1-', label = '$ss$')
    ax.plot(x, ss[0], 'C1--', label = '$ss(0)$')
    ax.plot(x, ss[-1],'C1--', label = '$ss(-1)$')
    ax.axhline(Q(i*skips*dt)/(2*Q0), 0, L)
    ax.set_xlim(0, L)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    title = 't=' + str(i*skips*dt)
    ax.set_title(title)
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=m//skips, interval=10, repeat=False)
plt.show()