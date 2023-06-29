#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:26:13 2023

@author: hugo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import integrate

from scipy.special import jv
from scipy.special import yv

from module_sol_v1 import goodrootfinder

pi = np.pi


Q, D, A, kappa, R, N = 2000, 20, 7500, 100, 7000, 30

f = 0.8
perc = 1/np.e

a = A/(pi*D)

r = np.linspace(a,R,1000)
t = np.linspace(0,1e5,1000+1)

def ss(r):
    o = Q/(kappa*pi*D)
    temp = (1-f)/(R**o - a**o)
    return temp*(r**o-R**o) + 1

def h1(r):
    o = 0.5*Q/(kappa*pi*D)
    temp = (1-f)/(R**o - a**o)
    return temp*(r**o-R**o) + 1 #- np.sin((r-a)/(R-a)*pi)/2

def h2(r):
    o = 2*Q/(kappa*pi*D)
    temp = (1-f)/(R**o - a**o)
    return temp*(r**o-R**o) + 1 #+ np.sin((r-a)/(R-a)*pi)/2

print('Looking for some roots...')
good_roots = goodrootfinder(N, 1e-5, 0.02, 200, Q, A, kappa, R, 0, D)
print('All set, continue integrating \n')

order = Q/(2*kappa*pi*D)
def phi_r(r,par):
    return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))

sign = [-np.sign(phi_r(0.01, good_roots[n]))/np.sqrt(integrate.quad(lambda r: phi_r(r,good_roots[n])**2, a, R)[0]) for n in range(N)]

def phi_n(r,n):
    return phi_r(r, good_roots[n]) * sign[n]

def labda_n(n):
    return good_roots[n][0]

def inner(n,m):
    return integrate.quad(lambda r: phi_n(r,m)*phi_n(r,n), a, R)[0]

G = np.array([[inner(n,m) for n in range(N)] for m in range(N)])

labda = np.array([labda_n(n) for n in range(N)])

inv = np.linalg.inv(G)

def solf(h):
    I1 = [sum([inv[n,j]*integrate.quad(lambda rho: (h(rho)-(1-f)/(R-a)*rho - (R*f-a)/(R-a))*phi_n(rho,j), a, R)[0] for j in range(N)]) for n in range(N)]
    I2 = [sum([inv[n,j]*integrate.quad(lambda r: 1/r*(kappa-Q/(pi*D))*(1-f)/(R-a)*phi_n(r,j), a, R)[0] for j in range(N)]) for n in range(N)]
    
    tempi1 = np.array([np.exp(-labda[n]*t) for n in range(N)])
    tempi2 = np.array([(1-np.exp(-labda[n]*t))/labda[n] for n in range(N)])
    
    T = np.zeros((len(t), N))
    
    for i in range(len(t)):
        T1 = tempi1[:,i] * I1
        T2 = tempi2[:,i] * I2
        T[i,:] = T1 + T2
      
    def ksi(r):
        return (1-f)/(R-a)*r + (R*f-a)/(R-a)
    
    KSI = np.zeros((len(t), len(r)))
    for i in range(len(t)):
        for j in range(len(r)):
            KSI[i,j] = ksi(r[j])
    
    sol = T @ np.array([phi_n(r,n) for n in range(N)]) + KSI

    s0 = h(r)
    s9 = ss(r)

    deltas = s9 - s0
    deltat = np.zeros(len(r))
    if np.sum(deltas)  < 0:
        for i in range(len(r)):
            for j in range(len(t)):
                if s9[i] - sol[j][i] >= deltas[i]*perc and deltat[i] == 0:
                    deltat[i] = t[j]
    else:
        for i in range(len(r)):
            for j in range(len(t)):
                if s9[i] - sol[j][i] <= deltas[i]*perc and deltat[i] == 0:
                    deltat[i] = t[j]
  
    dstot = 0
    Stot0 = integrate.quad(lambda r: h(r)*pi*r*D, a, R)[0]
    Stot9 = integrate.quad(lambda r: ss(r)*pi*r*D, a, R)[0]
    for j in range(len(t)):
        if abs(Stot9 - integrate.trapz(sol[j] * pi*D*r, r)) < perc*abs(Stot9 - Stot0) and dstot ==0:
            dstot = t[j]
            
    return sol, deltat, T, dstot


sol1, deltat1, T1, dstot1 = solf(h1)
sol2, deltat2, T2, dstot2 = solf(h2)

T_labda = -np.log(perc)/labda


fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(r/1000, deltat2/3600, 'C1-', label=r'$t_{ADJ}$ (increasing $s$)')

ax.hlines(T_labda[0]/3600, a/1000, R/1000, colors='k', ls='--', label = r'$1/ \lambda_1$')
#ax.hlines(L*A/Q, a, R, colors='r', label='$t_p$')
# ax.hlines(dstot2/3600, a/1000, R/1000, colors='C0', ls='--', label=r'$T_{ADJ}$ (increasing $S$)')

ax.set_xlim(a/1000,R/1000)
ax.set_ylim(0,max(np.max(deltat1),np.max(deltat2))/3600)
ax.set_xlabel(r'$r$ (km)')
ax.set_ylabel(r'$t$ (h)')
ax.legend()
plt.savefig("presentatie/results_sea_position_1.pdf")
plt.show()

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(r/1000, deltat2/3600, 'C1-', label=r'$t_{ADJ}$ (increasing $s$)')
ax.plot(r/1000, deltat1/3600, 'C0-', label=r'$t_{ADJ}$ (decreasing $s$)')
ax.hlines(T_labda[0]/3600, a/1000, R/1000, colors='k', ls='--', label = r'$1/ \lambda_1$')
#ax.hlines(L*A/Q, a, R, colors='r', label='$t_p$')
# ax.hlines(dstot2/3600, a/1000, R/1000, colors='C0', ls='--', label=r'$T_{ADJ}$ (increasing $S$)')
# ax.hlines(dstot1/3600, a/1000, R/1000, colors='C1', ls='--', label=r'$T_{ADJ}$ (increasing $S$)')
ax.set_xlim(a/1000,R/1000)
ax.set_ylim(0,max(np.max(deltat1),np.max(deltat2))/3600)
ax.set_xlabel(r'$r$ (km)')
ax.set_ylabel(r'$t$ (h)')
ax.legend()
plt.savefig("presentatie/results_sea_position_2.pdf")
plt.show()

# fig, axes = plt.subplots(1,2, figsize=(12,8))
# ax = axes[0]
# for j in range(10):
#     ax.plot(t, T1[:,j], 'C'+str(j), label='n='+str(int(j+1)))
# ax.set_xlim(0,t[-1])
# ax.set_ylim(-0.1,4.5)
# ax.set_xlabel(r'$t$ (s)')
# ax.set_ylabel(r'$T_n(t)$')
# #ax.set_yscale('log')
# ax.set_title('decreasing $s$')
# ax.legend()
# ax = axes[1]
# for j in range(10):
#     ax.plot(t, T2[:,j], 'C'+str(j), label='n='+str(int(j+1)))
# ax.set_xlim(0,t[-1])
# ax.set_ylim(-0.1,4.5)
# ax.set_xlabel(r'$t$ (s)')
# ax.set_ylabel(r'$T_n(t)$')
# ax.set_title('increasing $s$')
# #ax.set_yscale('log')
# ax.legend()
# plt.show()


n_skips =10
s01 = h1(r)
s02 = h2(r)
s9 = ss(r)
fig, ax = plt.subplots(1,1, figsize=(12,8))

ax.plot(r/1000, s02, 'C0--')
ax.plot(r/1000, s9, 'k-')
ax.set_xlim(a/1000,R/1000)
ax.set_ylim(0,1)
ax.set_xlabel(r'$r$ (km)')
ax.set_ylabel(r'$s/s_{sea}$')

# Plot the surface.
def init():
    ax.plot(r/1000, s02, label='oud evenwicht')
    ax.plot(r/1000, s9 - (s9-s02)/np.e, 'k:')
    ax.plot(r/1000, s9, label='nieuw evenwicht')
    ax.set_xlim(a/1000,R/1000)
    ax.set_ylim(0,1)
    ax.set_xlabel(r'$r$ (km)')
    ax.set_ylabel(r'$s/s_{sea}$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax.clear()
    ax.plot(r/1000, s02, 'C0--', label='oud evenwicht')
    ax.plot(r/1000, s9 - (s9-s02)/np.e, 'k:')
    ax.plot(r/1000, s9, 'k-', label='nieuw evenwicht')
    ax.plot(r/1000, sol2[i*n_skips], 'C0-')
    ax.set_xlim(a/1000,R/1000)
    ax.set_ylim(0,1)
    ax.set_xlabel(r'$r$ (km)')
    ax.set_ylabel(r'$s/s_{sea}$')
    ax.set_title(str(round(t[i]/3600,1)) + 'h')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=len(t)//n_skips, interval=10, repeat=False)
plt.show()

from matplotlib.animation import PillowWriter
#Save the animation as an animated GIF
plt.close()
# anim.save("presentatie/results_sea.gif", dpi=120,
#           writer=PillowWriter(fps=10))


# fig, ax = plt.subplots(1,1, figsize=(8,8))
# ax.plot(r, s01, 'k--', label=r'$s(t=0)$')
# ax.plot(r, s9 - (s9-s01)/np.e, 'k:', label = r'$s(t=t_{ADJ})$')
# ax.plot(r, s9, 'k-', label=r'$s(t = \infty)$')
# ax.plot(r, sol1[int(len(t)//10)], 'C0-', label = '$s(t$ = ' + str(int(t[len(t)//10])) + ' s)')
# ax.plot(r, sol1[int(3*len(t)//8)], 'C1-', label = '$s(t$ = ' + str(int(t[int(3*len(t)//8)])) + ' s)')
# ax.plot(r, sol1[int(3.5*len(t)//4)], 'C2-', label = '$s(t$ = ' + str(int(t[int(3.5*len(t)//4)])) + ' s)')
# ax.set_xlim(a,R)
# ax.set_ylim(f, 1)
# ax.set_xlabel(r'$r$ (m)')
# ax.set_ylabel(r'$s/s_{sea}$')
# ax.legend()
# plt.show()