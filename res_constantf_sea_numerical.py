#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 19:09:18 2023

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


Q, D, A, kappa, R, N = 200, 20, 7500, 180, 7000, 30

print(Q/(kappa*pi*D))

f = 0.5
perc = 0.1

a = A/(pi*D)

r = np.linspace(a,R,1000)
t = np.linspace(0,1e6,1000)

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
good_roots = goodrootfinder(N, 1e-5, 0.04, 1000, Q, A, kappa, R, 0, D)
print('All set, continue integrating \n')

order = Q/(2*kappa*pi*D)
def phi_r(r,par):
    return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))

sign = [np.sign(phi_r(0.01, good_roots[n]))/np.sqrt(integrate.quad(lambda r: phi_r(r,good_roots[n])**2, a, R)[0]) for n in range(N)]

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

def fd(Q, A, kappa, R, D, n, n_steps, dt, h):
    print('Finite differences started')    
    a = A/(pi*D)

    dr = (R-a)/(n+1)
    
    M = np.diag(-2*np.ones(n)) + np.diag(np.ones(n-1), k=-1) + np.diag(np.ones(n-1), k=1)
    M = kappa/(dr**2)*M
    
    b = np.zeros(n)
    b[0] = f*kappa/dr**2
    b[n-1] = kappa/dr**2
    
    K = np.diag(-np.ones(n), k=0) + np.diag( np.ones(n-1), k=1)
    for j in range(0,n):
        K[j,:] = 1/(a+(j)*dr) * (kappa-Q/(pi*D)) * 1/(dr)*K[j,:]
    
    d = np.zeros(n)
    d[n-1] = 1/(R-dr) * (kappa-Q/(pi*D)) * 1/(dr)
    
    
    P = M + K
    w = b + d
        
    I = np.diag(np.ones(n))
    J = np.linalg.inv(I-dt/2*P)
    T = J @ (I + dt/2*P)
    F = dt*J@w

    res = [np.zeros(n) for i in range(n_steps)]
    res[0] = h(r_num)
    
    sr = np.zeros(n_steps)
    sr[0] = integrate.trapz(h(r_num)*pi*r_num*D)
    
    #T = dia_matrix(T)
    
    for i in range(1,n_steps):
        if i*10 % n_steps == 0:
            print('integration @ ' + str(i*100//n_steps), '%')
        res[i] = T @ res[i-1] + F
        
        sr[i] = integrate.trapz(res[i]*pi*r_num*D)
    
    s0 = h(r_num)
    s9 = ss(r_num)

    deltas = s9 - s0
    deltat = np.zeros(len(r_num))
    if np.sum(deltas)  < 0:
        for i in range(len(r_num)):
            for j in range(1,n_steps,1):
                if s9[i] - res[j][i] >= deltas[i]*perc and deltat[i] == 0:
                    deltat[i] = dt*j
    else:
        for i in range(len(r_num)):
            for j in range(1,n_steps,1):
                if s9[i] - res[j][i] <= deltas[i]*perc and deltat[i] == 0:
                    deltat[i] = dt*j
    dstot = 0
    Stot0 = integrate.trapz(h(r_num)*pi*r_num*D) #integrate.quad(lambda r: h(r)*pi*r*D, a, R)[0]
    Stot9 = integrate.trapz(ss(r_num)*pi*r_num*D) #integrate.quad(lambda r: ss(r)*pi*r*D, a, R)[0]
    for j in range(1,n_steps):
        if abs(Stot9 - sr[j]) < perc*abs(Stot9 - Stot0) and dstot == 0:
            print('het is gebeurt')
            dstot = dt*j
    print('Finite differences finished\n')    
    return res, deltat, dstot

n = 6000
r_num = np.linspace(a,R,n+2)[1:-1]
dt = 1000
n_steps = int(t[-1]/dt)

fd_res1, fd_deltat1, fd_dstot1 = fd(Q, A, kappa, R, D, n, n_steps, dt, h1)
fd_res2, fd_deltat2, fd_dstot2 = fd(Q, A, kappa, R, D, n, n_steps, dt, h2)

n_skips = int((t[1]-t[0])//dt)


# fig, ax = plt.subplots(1,1, figsize=(8,8))
# for j in range(N):
#     ax.plot(r, phi_n(r,j))
# ax.set_xlim(a,R)
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'e-folding $t$')
# ax.legend()
# plt.show()


fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(r, deltat1, 'C0.', ms=4,label=r'$t_{ADJ}$ decrease, eigen expansion')
ax.plot(r, deltat2, 'C1.', ms=4, label=r'$t_{ADJ}$ increase, eigen expansion')
ax.plot(r_num, fd_deltat1, 'C2.', ms= 4, label=r'$t_{ADJ}$ decrease, numerical')
ax.plot(r_num, fd_deltat2, 'C3.', ms= 4,label=r'$t_{ADJ}$ increase, numerical')
ax.hlines(T_labda[0], a, R, colors='k', label = r'$1/ \lambda_1$')
#ax.hlines(L*A/Q, a, R, colors='r', label='$t_p$')
ax.hlines(dstot1, a, R, colors='C0', label=r'$T_{ADJ}$ decrease, eigen expansion')
ax.hlines(dstot2, a, R, colors='C1', label=r'$T_{ADJ}$ increase, eigen expansion')
ax.hlines(fd_dstot1, a, R, colors='C2', ls='-', label=r'$T_{ADJ}$ decrease, numerical')
ax.hlines(fd_dstot2, a, R, colors='C3', ls='-', label=r'$T_{ADJ}$ increase, numerical')
ax.set_xlim(a,R)
ax.set_ylim(0,150000)
ax.set_xlabel(r'$r$ (m)')
ax.set_ylabel(r'$t$ (s)')
ax.legend()
plt.show()



s01 = h1(r)
s02 = h2(r)
s9 = ss(r)
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(r, s01, 'C0--')
ax.plot(r, s02, 'C1--')
ax.plot(r, s9, 'k-')
ax.set_xlim(a,R)
ax.set_ylim(0,1)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$s$')

# Plot the surface.
def init():
    ax.plot(r, s01, label='initial1')
    ax.plot(r, s02, label='initial2')
    ax.plot(r, s9, label='final')
    ax.set_xlim(a,R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax.clear()
    ax.plot(r, s01, 'C0--', label='initial1')
    ax.plot(r, s02, 'C1--', label='initial2')
    ax.plot(r, s9 - (s9-s01)/np.e, 'k:')
    ax.plot(r, s9 - (s9-s02)/np.e, 'k:')
    ax.plot(r, s9, 'k-', label='final')
    ax.plot(r, sol1[i], 'C0-')
    ax.plot(r, sol2[i], 'C1-')
    ax.plot(r_num, fd_res1[i*n_skips], 'C2-')
    ax.plot(r_num, fd_res2[i*n_skips], 'C3-')
    ax.set_xlim(a,R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.set_title(str(t[i]))
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=len(t), interval=10, repeat=False)
plt.show()
