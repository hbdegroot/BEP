#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:50:21 2023

@author: hugo
"""
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


Pe = 10

f = 0.5
perc = 1/np.e

xi = np.linspace(0,1,200)
tau = np.linspace(0,1,200)

N = 30

def ss(xi):
    temp = np.exp(-Pe)
    return f/(1-temp)*(np.exp(-Pe*xi) - temp)

def h1(xi):
    P = Pe*0.5
    temp = np.exp(-P)
    return f/(1-temp)*(np.exp(-P*xi) - temp)

def h2(xi):
    P = Pe*2
    temp = np.exp(-P)
    return f/(1-temp)*(np.exp(-P*xi) - temp)

def phi_n(xi,n):
    n+=1
    return np.exp(-Pe*xi/2)*np.sin(n*pi*xi)


def labda_n(n):
    n+=1
    return Pe**2/4 + n**2 * pi**2

def inner(n,m):
    return integrate.quad(lambda xi: phi_n(xi,m)*phi_n(xi,n), 0, 1)[0]

G = np.array([[inner(n,m) for n in range(N)] for m in range(N)])

labda = np.array([labda_n(n) for n in range(N)])

inv = np.linalg.inv(G)

def solf(h):
    I1 = [sum([inv[n,j]*integrate.quad(lambda x: (h(x)-f*(1-x))*phi_n(x,j), 0, 1)[0] for j in range(N)]) for n in range(N)]
    I2 = [sum([inv[n,j]*integrate.quad(lambda x: (-Pe*f)*phi_n(x,j), 0, 1)[0] for j in range(N)]) for n in range(N)]
    
    tempi1 = np.array([np.exp(-labda[n]*tau) for n in range(N)])
    tempi2 = np.array([(1-np.exp(-labda[n]*tau))/labda[n] for n in range(N)])
    
    T = np.zeros((len(tau), N))
    
    for i in range(len(tau)):
        T1 = tempi1[:,i] * I1
        T2 = tempi2[:,i] * I2
        T[i,:] = T1 + T2
      
    def ksi(xi):
        return f*(1-xi)
    
    KSI = np.zeros((len(tau), len(xi)))
    for i in range(len(tau)):
        for j in range(len(xi)):
            KSI[i,j] = ksi(xi[j])
    
    sol = T @ np.array([phi_n(xi,n) for n in range(N)]) + KSI

    s0 = h(xi)
    s9 = ss(xi)

    deltas = s9 - s0
    deltat = np.zeros(len(xi))
    if np.sum(deltas)  < 0:
        for i in range(len(xi)):
            for j in range(len(tau)):
                if s9[i] - sol[j][i] >= deltas[i]*perc and deltat[i] == 0:
                    deltat[i] = tau[j]
    else:
        for i in range(len(xi)):
            for j in range(len(tau)):
                if s9[i] - sol[j][i] <= deltas[i]*perc and deltat[i] == 0:
                    deltat[i] = tau[j]
  
    dstot = 0
    Stot0 = integrate.quad(h, 0, 1)[0]
    Stot9 = integrate.quad(ss, 0, 1)[0]
    for j in range(len(tau)):
        if abs(Stot9 - integrate.trapz(sol[j], xi)) < perc*abs(Stot9 - Stot0) and dstot ==0:
            dstot = tau[j]
            
    return sol, deltat, T, dstot


sol1, deltat1, T1, dstot1 = solf(h1)
sol2, deltat2, T2, dstot2 = solf(h2)

T_labda = -np.log(perc)/labda

 


fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(xi, deltat1, 'C0.', label='salinity (down)')
ax.plot(xi, deltat2, 'C1.', label='salinity (up)')
ax.hlines(T_labda, 0, 1, colors='k', label = r'$1/ \lambda$')

ax.hlines(dstot1, 0, 1, colors='C0', label='total salt (down)')
ax.hlines(dstot2, 0, 1, colors='C1', label='total salt (up)')
ax.set_xlim(0,1)
ax.set_xlabel(r'$xi$ (m)')
ax.set_ylabel(r'e-folding $tau$ (s)')
ax.legend()
plt.show()



s01 = h1(xi)
s02 = h2(xi)
s9 = ss(xi)
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(xi, s01, 'C0--')
ax.plot(xi, s02, 'C1--')
ax.plot(xi, s9, 'k-')
ax.set_xlim(0,1)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$s$')

# Plot the surface.
def init():
    ax.plot(xi, s01, label='initial1')
    ax.plot(xi, s02, label='initial2')
    ax.plot(xi, s9, label='final')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel(r'$xi$')
    ax.set_ylabel(r'$\zeta$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax.clear()
    ax.plot(xi, s01, 'C0--', label='initial1')
    ax.plot(xi, s02, 'C1--', label='initial2')
    ax.plot(xi, s9 - (s9-s01)/np.e, 'k:')
    ax.plot(xi, s9 - (s9-s02)/np.e, 'k:')
    ax.plot(xi, s9, 'k-', label='final')
    ax.plot(xi, sol1[i], 'C0-')
    ax.plot(xi, sol2[i], 'C1-')
    ax.set_xlim(0,1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$xi$')
    ax.set_ylabel(r'$\zeta$')
    ax.set_title(str(tau[i]))
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=len(tau), interval=10, repeat=False)
plt.show()
