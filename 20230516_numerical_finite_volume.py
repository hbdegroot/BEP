#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:54:21 2023

@author: hugo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from numpy import pi

Q, A, k, kappa, R, L, D = 150, 1000, 400, 400, 20000, 10000, 2

a = A/(pi*D)

nx = 20
x = np.linspace(0,L,nx+2)[:-1]
dx = x[1]-x[0]
print(x)

nr = 20
r = np.linspace(0,R,nr+2)[1:-1]
dr = r[1]-r[0]
print(r)

N = nx + 1 + nr

n = nx

n_steps = int(1e5)
dt = 1

def rj(j):
    #returns r < r_j
    return dr*(j-n-1/2)

M = np.diag(-np.ones(N)) + np.diag(np.ones(N-1), k=-1)
M[:n,:] = Q/(A*dx)*M[:n,:]

M[n, :] = Q/(pi*dr**2*D/4) * M[n, :]

for j in range(n+1,N):
    M[j, :] = Q/(pi*(rj(j+1)**2-rj(j)**2)/2*D) * M[j, :]
    
K = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), k=-1) + np.diag(np.ones(N-1), k=1)
K[:n,:] = k/dx**2*K[:n,:]

K[n, n-1] = k*A/dx / (A*dx/2 + pi*dr**2*D/4)
K[n, n] = -(k*A/dx + kappa*pi*D/2) / (A*dx/2 + pi*dr**2*D/4)
K[n, n+1] = kappa*pi/2*D / (A*dx/2 + pi*dr**2*D/4)

for j in range(n+1,N):
    K[j, j-1] = kappa*pi*rj(j)*D / (pi*(rj(j+1)**2-rj(j)**2)/2*D)
    K[j, j] = -kappa*pi*(rj(j)+rj(j+1))*D / (pi*(rj(j+1)**2-rj(j)**2)/2*D)
    if j+1 < N:
        K[j, j+1] = kappa*pi*rj(j+1)*D / (pi*(rj(j+1)**2-rj(j)**2)/2*D)

f = np.zeros(N)
f[N-1] = kappa*pi*rj(N)*D / (pi*(rj(N)**2-rj(N-1)**2)/2*D)

P = M + K

I = np.diag(np.ones(N))
J = np.linalg.inv(I-dt/2*P)
T = J @ (I + dt/2*P)
F = dt*J@f

    
def ssx(x):
    pwr = Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
    c1 = d1*a**pwr
    c2 = -np.exp(-Q*L/(k*A))*c1
    return c1 * np.exp(-Q*x/(k*A)) + c2

def ssr(r):
    pwr = Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
    d2 = 1-d1*R**pwr
    return d1*r**pwr + d2

def ssx0(x):
    pwr = 2*Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-2*Q*L/(k*A)))
    c1 = d1*a**pwr
    c2 = -np.exp(-2*Q*L/(k*A))*c1
    return c1 * np.exp(-2*Q*x/(k*A)) + c2

def ssr0(r):
    pwr = 2*Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-2*Q*L/(k*A)))
    d2 = 1-d1*R**pwr
    return d1*r**pwr + d2

t = np.arange(0,dt*(n_steps+1), dt)

res = [np.zeros(N) for i in range(n_steps)]
res[0] = np.append(ssx0(x)[::-1], ssr0(r))

for i in range(1,n_steps):
    if i*10 % n_steps == 0:
        print('integration @ ' + str(i*100//n_steps), '%')
    res[i] = T @ res[i-1] + F
    #res[i] = res[i-1] + dt*(P @ res[i-1] + F)
    

x_ext = np.append(x, np.array([L]))
r_ext0 = np.append(np.array([a]), r)
r_ext = np.append(r_ext0, np.array([R]))
axs_ext = np.append(-x_ext[::-1], r_ext)
axs = np.append(-x[::-1], r)
s0 = np.append(ssx0(x_ext)[::-1], ssr0(r_ext))
sn = np.append(ssx(x_ext)[::-1], ssr(r_ext))


n_skips = n_steps//100
fig, axes = plt.subplots(2,1, figsize=(8,8))

ax = axes[0]
ax.plot(axs_ext, s0)
ax.plot(axs_ext, sn)
ax.set_xlim(-L, R)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'$s$')

ax = axes[1]
ax.set_xlim(0, t[-1])
# ax.set_ylim(0, ylimit)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$ds/dx & ds/dr$')

# Plot the surface.
def init():
    ax = axes[0]
    ax.plot(axs_ext, s0, label='initial')
    ax.plot(axs_ext, sn, label='final')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    
    ax = axes[1]
    ax.set_xlim(0, t[-1])
    # ax.set_ylim(0, ylimit)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    return 

# animation function.  This is called sequentially
def animate(i):
    i = i*n_skips
    ax = axes[0]
    ax.clear()
    ax.plot(axs_ext, s0, label='initial')
    ax.plot(axs_ext, sn, label='final')
    ax.plot(axs, res[i], label=str(t[i]))
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.legend()
    
    ax = axes[1]
    ax.clear()
    ax.plot(axs, axs, label='label')
    ax.set_xlim(0, t[-1])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=n_steps//n_skips, interval=10, repeat=False)
plt.show()
