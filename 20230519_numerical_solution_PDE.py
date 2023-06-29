#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 08:50:10 2023

@author: hugo
"""
import numpy as np
from numpy import pi

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation     


Q, A, k, kappa, R, L, D = 150, 2000, 400, 400, 10000, 10000, 10
n, m, n_steps, dt = 100, 100, int(1e4), 1

t = np.arange(0,(n_steps+1)*dt, dt)

a = A/(pi*D)

N = n + m

dx = L/(n+1)
dr = (R-a)/(m+1)

x = np.linspace(0,L,n+2)[1:-1]
r = np.linspace(a,R,m+2)[1:-1]

alpha = (kappa/dr) / (kappa/dr + k/dx)
beta = (k/dx) / (kappa/dr + k/dx)


M = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), k=-1) + np.diag(np.ones(N-1), k=1)
M[n-1, n-1] = -2 + beta
M[n-1, n] = alpha
M[:n,:] = k/(dx**2)*M[:n,:]

M[n, n] = -2 + alpha
M[n, n-1] = beta
M[n:,:] = kappa/(dr**2)*M[n:,:]

b = np.zeros(N)
b[N-1] = kappa/dr**2

K = np.diag(np.append(np.ones(n), -np.ones(m)), k=0) + np.diag(np.append(-np.ones(n-1), np.zeros(m)), k=-1) + np.diag(np.append(np.zeros(n), np.ones(m-1)), k=1)
K[:n,:] = -(Q/A)/(dx)*K[:n,:]
for j in range(n,N):
    K[j,:] = 1/(a+(j-n+1)*dr) * (kappa-Q/(pi*D)) * 1/(dr)*K[j,:]

d = np.zeros(N)
d[N-1] = 1/(R-dr) * (kappa-Q/(pi*D)) * 1/(dr)

P = M + K
f = b + d

I = np.diag(np.ones(N))
J = np.linalg.inv(I-dt/2*P)
T = J @ (I + dt/2*P)
F = dt*J@f

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

res = [np.zeros(N) for i in range(n_steps)]
res[0] = np.append(ssx0(x)[::-1], ssr0(r))

dsdx = np.zeros(n_steps)
dsdr = np.zeros(n_steps)

for i in range(1,n_steps):
    if i*10 % n_steps == 0:
        print('integration @ ' + str(i*100//n_steps), '%')
    res[i] = T @ res[i-1] + F
    
    dsdx[i] =  (-alpha*res[i][n] + (1-beta)*res[i][n-1])/dx
    dsdr[i] =  (-beta*res[i][n-1] + (1-alpha)*res[i][n])/dr
    #res[i] = res[i-1] + dt*(P @ res[i-1] + f)

axs = np.append(-x[::-1], r)
s0 = np.append(ssx0(x)[::-1], ssr0(r))
sn = np.append(ssx(x)[::-1], ssr(r))


n_skips = n_steps//100

fig, axes = plt.subplots(2,1, figsize=(8,8))

ax = axes[0]
ax.plot(axs, s0)
ax.plot(axs, sn)
ax.plot(axs, res[0])
ax.set_xlim(-L, R)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'$s$')

ax = axes[1]
ax.plot(t[0], -k*dsdx[0], label='k s_x')
ax.plot(t[0], kappa*dsdr[0], '--', label='K s_r')
ax.axhline(Q/A*res[0][n], label='Q/A s(0)')
ax.set_xlim(0, t[-1])
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$ds/dx & ds/dr$')

plt.show()

# Plot the surface.
def init():
    ax = axes[0]
    ax.plot(axs, s0, label='initial')
    ax.plot(axs, sn, label='final')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    
    ax = axes[1]
    ax.plot(t[0], -k*dsdx[0], label='k s_x')
    ax.plot(t[0], kappa*dsdr[0], '--', label='K s_r')
    ax.axhline(Q/A*res[0][n], label='Q/A s(0)')
    ax.set_xlim(0, t[-1])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax = axes[0]
    ax.clear()
    ax.plot(axs, s0, label='initial')
    ax.plot(axs, sn, label='final')
    ax.plot(axs, res[i*n_skips], label = 'numerical scheme')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.set_title('t=' + str(t[i*n_skips]))
    ax.legend()
    
    ax = axes[1]
    ax.clear()    
    ax.plot(t[:i*n_skips], -k*dsdx[:i*n_skips], label='k s_x')
    ax.plot(t[:i*n_skips], kappa*dsdr[:i*n_skips], '--', label='K s_r')
    ax.axhline(Q/A*res[i*n_skips][n], label='Q/A s(0)')
    ax.set_xlim(0, t[-1])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    return 

anim = FuncAnimation(fig, animate, frames=n_steps//n_skips, interval=10, repeat=False)
plt.show()