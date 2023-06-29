#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:05:02 2023

@author: hugo
"""
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
from scipy.optimize import root

pi = np.pi

N = 30
q = 0.02
P = 0.001

f = 0.8
perc = 1/np.e

rho = np.linspace(0,1,400)
tau = np.linspace(0,10,200)

def ss(rho):
    temp = ( (rho+q)**P - q**P )/( (1+q)**P - q**P ) 
    return temp*(1-f) + f

def h1(rho):
    m = 1/1.4
    temp = ( (rho+q)**(P*m) - q**(P*m) )/( (1+q)**(P*m) - q**(P*m) ) 
    return temp*(1-f) + f

def h2(rho):
    m = 1.4
    temp = ( (rho+q)**(P*m) - q**(P*m) )/( (1+q)**(P*m) - q**(P*m) ) 
    return temp*(1-f) + f

def phi_r(rho,P,q,par):
    return (rho+q)**(P/2)*jv(P/2, np.sqrt(par[0])*(rho+q)) + par[1]*(rho+q)**(P/2)*yv(P/2, np.sqrt(par[0])*(rho+q))
  
r = np.linspace(0,1,500)

def good_root_finder(N,P,q):
    phi_bc0 = lambda rt: phi_r(0,P,q,rt)
    phi_bc1 = lambda rt: phi_r(1,P,q,rt)
         
    def func_r(rt):
        return phi_bc0(rt),phi_bc1(rt)
        
    def n_roots(rt):
        temp = 0
        if abs(phi_r(0,P,q,rt))>0.001: return -1
        elif abs(phi_r(1,P,q,rt))>0.001: return -1
        for k in range(5,len(r)-5):
            if phi_r(r[k],P,q,rt)*phi_r(r[k-1],P,q,rt)<0:
                temp+=1
        return temp
    
    def finder(l0,l1,n):
        for i in np.linspace(l0,l1,10):
             for j in np.linspace(-10,10,6):
                 rt = root(func_r, x0 =(i,j)).x
                 M = n_roots(rt)
                 #print(rt)
                 if M == n: return rt[0], rt[1]
                     
                 elif M>n and rt[0]<l1: return finder(l0/2,rt[0],n)
        return finder(l1,l1*2,n)
    
    roots = [np.zeros(2) for n in range(N)]
    for n in range(N):
        roots[n] = finder(roots[n-1][0],roots[n-1][0]*1.5+1e-1,n)                                                                                      
    return roots

print('Looking for some roots...')
good_roots = good_root_finder(N,P,q)
print('They\'re  here!\nContinue integrating...\n')
sign = [np.sign(phi_r(0.01, P, q, good_roots[n]))/np.sqrt(integrate.quad(lambda r: phi_r(r, P, q, good_roots[n])**2, 0, 1)[0]) for n in range(N)]

def phi_n(r,n):
    return phi_r(r, P, q, good_roots[n]) * sign[n]

def labda_n(n):
    return good_roots[n][0]

def inner(n,m):
    return integrate.quad(lambda r: phi_n(r,m)*phi_n(r,n), 0, 1)[0]

G = np.array([[inner(n,m) for n in range(N)] for m in range(N)])

labda = np.array([labda_n(n) for n in range(N)])

inv = np.linalg.inv(G)

def solf(h):
    I1 = [sum([inv[n,j]*integrate.quad(lambda rho: (h(rho)-f-rho*(1-f))*phi_n(rho,j), 0, 1)[0] for j in range(N)]) for n in range(N)]
    I2 = [sum([inv[n,j]*integrate.quad(lambda rho: (1-P)/(rho+q)*(1-f)*phi_n(rho,j), 0, 1)[0] for j in range(N)]) for n in range(N)]
    
    tempi1 = np.array([np.exp(-labda[n]*tau) for n in range(N)])
    tempi2 = np.array([(1-np.exp(-labda[n]*tau))/labda[n] for n in range(N)])
    
    T = np.zeros((len(tau), N))
    
    for i in range(len(tau)):
        T1 = tempi1[:,i] * I1
        T2 = tempi2[:,i] * I2
        T[i,:] = T1 + T2
      
    def ksi(rho):
        return f + (1-f)*rho
    
    KSI = np.zeros((len(tau), len(rho)))
    for i in range(len(tau)):
        for j in range(len(rho)):
            KSI[i,j] = ksi(rho[j])
    
    sol = T @ np.array([phi_n(rho,n) for n in range(N)]) + KSI

    s0 = h(rho)
    s9 = ss(rho)

    deltas = s9 - s0
    deltat = np.zeros(len(rho))
    if np.sum(deltas)  < 0:
        for i in range(len(rho)):
            for j in range(len(tau)):
                if s9[i] - sol[j][i] >= deltas[i]*perc and deltat[i] == 0:
                    deltat[i] = tau[j]
    else:
        for i in range(len(rho)):
            for j in range(len(tau)):
                if s9[i] - sol[j][i] <= deltas[i]*perc and deltat[i] == 0:
                    deltat[i] = tau[j]
  
    dstot = 0
    Stot0 = integrate.quad(lambda rho: h(rho)*pi*(rho+q), 0, 1)[0]
    Stot9 = integrate.quad(lambda rho: ss(rho)*pi*(rho+q), 0, 1)[0]
    for j in range(len(tau)):
        if abs(Stot9 - integrate.trapz(sol[j]*pi*(rho+q), rho)) < perc*abs(Stot9 - Stot0) and dstot ==0:
            dstot = tau[j]
            
    return sol, deltat, T, dstot


sol1, deltat1, T1, dstot1 = solf(h1)
sol2, deltat2, T2, dstot2 = solf(h2)

T_labda = -np.log(perc)/labda



# fig, ax = plt.subplots(1,1, figsize=(8,8))
# for j in range(N):
#     ax.plot(rho, phi_n(rho,j))
# ax.set_xlim(0,1)
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'e-folding $t$')
# ax.legend()
# plt.show()


fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(rho, deltat1, 'C0.', label='result')
ax.plot(rho, deltat2, 'C1.', label='result')

ax.hlines(T_labda, 0, 1, colors='k', label = r'$1/ \lambda$')
#ax.hlines(L*A/Q, a, R, colors='r', label='$t_p$')
ax.hlines(dstot1, 0, 1, colors='C0', label='$dstot1$')
ax.hlines(dstot2, 0, 1, colors='C1', label='$dstot2$')
ax.set_xlim(0,1)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'e-folding $t$')
ax.legend()
plt.show()



s01 = h1(rho)
s02 = h2(rho)
s9 = ss(rho)
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(rho, s01, 'C0--')
ax.plot(rho, s02, 'C1--')
ax.plot(rho, s9, 'k-')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$s$')

# Plot the surface.
def init():
    ax.plot(rho, s01, label='initial1')
    ax.plot(rho, s02, label='initial2')
    ax.plot(rho, s9, label='final')
    ax.set_xlim(0,1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$s$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax.clear()
    ax.plot(rho, s01, 'C0--', label='initial1')
    ax.plot(rho, s02, 'C1--', label='initial2')
    ax.plot(rho, s9 - (s9-s01)/np.e, 'k:')
    ax.plot(rho, s9 - (s9-s02)/np.e, 'k:')
    ax.plot(rho, s9, 'k-', label='final')
    ax.plot(rho, sol1[i], 'C0-')
    ax.plot(rho, sol2[i], 'C1-')
    ax.set_xlim(0,1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.set_title(str(tau[i]))
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=len(tau), interval=10, repeat=False)
plt.show()
