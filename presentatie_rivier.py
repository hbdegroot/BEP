#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:22:09 2023

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
ax.plot(-x/1000, deltat2/3600, 'C1-', label=r'$t_{ADJ}$ (increasing $s$)')
ax.hlines(T_labda[0]/3600, -L/1000, 0, colors='k', ls='--', label = r'$1/ \lambda_1$')
# ax.hlines(dstot2/3600, -L/1000, 0, colors='C1', ls='--', label=r'$T_{ADJ}$ (increasing $S$)')
ax.set_xlim(-L/1000,0)
ax.set_ylim(0,275000/3600)
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$t$ (h)')
ax.legend()
plt.savefig("presentatie/results_river_position_1.pdf")
plt.show()

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(-x/1000, deltat1/3600, 'C0-', label=r'$t_{ADJ}$ (decreasing $s$)')
ax.plot(-x/1000, deltat2/3600, 'C1-', label=r'$t_{ADJ}$ (increasing $s$)')
ax.hlines(T_labda[0]/3600, -L/1000, 0, colors='k', ls='--', label = r'$1/ \lambda_1$')
#ax.hlines(L*A/Q, 0, L, colors='r', label='$t_p$')
#ax.hlines(0.5*L*A/Q, 0, L, colors='r', label='0.5$t_p$')
# ax.hlines(dstot1/3600, -L/1000, 0, colors='C0', ls='--', label=r'$T_{ADJ}$ (decreasing $S$)')
# ax.hlines(dstot2/3600, -L/1000, 0, colors='C1', ls='--', label=r'$T_{ADJ}$ (increasing $S$)')
ax.set_xlim(-L/1000,0)
ax.set_ylim(0,275000/3600)
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$t$ (h)')
ax.legend()
plt.savefig("presentatie/results_river_position_2.pdf")
plt.show()


# fig, axes = plt.subplots(1,2, figsize=(12,8))
# ax = axes[0]
# for j in range(10):
#     ax.plot(t, T1[:,j], 'C'+str(j), label='n='+str(int(j+1)))
# ax.set_xlim(0,1e6)
# ax.set_xlabel(r'$t$ (s)')
# ax.set_ylabel(r'$T_n(t)$')
# #ax.set_yscale('log')
# ax.legend()

# ax = axes[1]
# for j in range(10):
#     ax.plot(t, T1[:,j]-T1[-1,j])
# ax.set_xlim(0,1e6)
# ax.set_xlabel(r'$t$ (s)')
# ax.set_ylabel(r'$T_n(t)-T_n(\infty)$')
# #ax.set_yscale('log')
# plt.show()

# fig, axes = plt.subplots(1,2, figsize=(12,8))
# ax = axes[0]
# for j in range(10):
#     ax.plot(t, T2[:,j], 'C'+str(j), label='n='+str(int(j+1)))
# ax.set_xlim(0,1e6)
# ax.set_xlabel(r'$t$ (s)')
# ax.set_ylabel(r'$T_n(t)$')
# #ax.set_yscale('log')
# ax.legend()

# ax = axes[1]
# for j in range(10):
#     ax.plot(t, T2[:,j]-T2[-1,j])
# ax.set_xlim(0,1e6)
# ax.set_xlabel(r'$t$ (s)')
# ax.set_ylabel(r'$T_n(t)-T_n(\infty)$')
# #ax.set_yscale('log')
# #ax.legend()

plt.show()


# fig, axes = plt.subplots(1,2, figsize=(12,8))
# ax = axes[0]
# for j in range(10):
#     ax.plot(t, T1[:,j], 'C'+str(j), label='n='+str(int(j+1)))
# ax.set_xlim(0,1e6)
# ax.set_ylim(-1, 0.1)
# ax.set_xlabel(r'$t$ (s)')
# ax.set_ylabel(r'$T_n(t)$')
# ax.set_title('decreasing $s$')
# #ax.set_yscale('log')
# ax.legend()
# ax = axes[1]
# for j in range(10):
#     ax.plot(t, T2[:,j], 'C'+str(j), label='n='+str(int(j+1)))
# ax.set_xlim(0,1e6)
# ax.set_ylim(-1, 0.1)
# ax.set_xlabel(r'$t$ (s)')
# ax.set_ylabel(r'$T_n(t)$')
# ax.set_title('increasing $s$')
# #ax.set_yscale('log')
# ax.legend()
# plt.show()


s01 = h1(x)
s02 = h2(x)
s9 = ss(x)

n_skips = 10

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.plot(-x/1000, s02, 'C0--')
ax.plot(-x/1000, s9 - (s9-s02)/np.e, 'k:')
ax.plot(-x/1000, s9, 'k-')
ax.set_xlim(-L/1000,0)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$s/s_{sea}$')

# Plot the surface.
def init():
    ax.plot(-x/1000, s02, label='oud evenwicht')
    ax.plot(-x/1000, s9, label='nieuw evenwicht')
    ax.set_xlim(-L/1000,0)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$s/s_{sea}$')
    ax.set_title(str(round(t[0]/3600),1) + 'h')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax.clear()
    ax.plot(-x/1000, s02, 'C0--', label='oud evenwicht')
    ax.plot(-x/1000, s9 - (s9-s02)/np.e, 'k:')
    ax.plot(-x/1000, s9, 'k-', label='nieuw evenwicht')
    ax.plot(-x/1000, sol2[i*n_skips], 'C0-')
    ax.set_xlim(-L/1000,0)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$s/s_{sea}$')
    ax.set_title(str(round(t[i*n_skips]/3600,1)) + 'h')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=len(t)//n_skips, interval=10, repeat=False)
plt.show()

from matplotlib.animation import PillowWriter
#Save the animation as an animated GIF
plt.close()
# anim.save("presentatie/results_river.gif", dpi=120,
#           writer=PillowWriter(fps=10))

