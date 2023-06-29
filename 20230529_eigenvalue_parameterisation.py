#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 19:31:46 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt

from numpy import pi
from scipy import integrate
from scipy.special import jv
from scipy.special import yv
from scipy.optimize import root


Np = 51
Nq = 51

labda = np.zeros((Np,Nq))
c = np.zeros((Np,Nq))

P = np.linspace(0.001,0.5,Np)
q = np.linspace(0.001,0.1,Nq)

def phi_r(rho,P,q,par):
    return (rho+q)**(P/2)*jv(P/2, np.sqrt(par[0])*(rho+q)) + par[1]*(rho+q)**(P/2)*yv(P/2, np.sqrt(par[0])*(rho+q))
  
r = np.linspace(0,1,500)
        
for i in range(Np): 
    for j in range(Nq):                                                                                                
        phi_bc0 = lambda rt: phi_r(0,P[i],q[j],rt)
        phi_bc1 = lambda rt: phi_r(1,P[i],q[j],rt)
         
        def func_r(rt):
             return phi_bc0(rt),phi_bc1(rt)
        
        def n_roots(rt):
             temp = 0
             if abs(phi_r(0,P[i],q[j],rt))>0.01: return -1
             elif abs(phi_r(1,P[i],q[j],rt))>0.01: return -1
             for k in range(10,len(r)-20):
                 if phi_r(r[k],P[i],q[j],rt)*phi_r(r[k-1],P[i],q[j],rt)<0:
                     temp+=1
             return temp
        
        def finder(l0,l1):
            for i in np.linspace(l0,l1,10):
                 for j in np.linspace(-10,10,6):
                     rt = root(func_r, x0 =(i,j)).x
                     N = n_roots(rt)
                     #print(rt)
                     if N == 0: return rt[0], rt[1]
                         
                     elif N>0 and rt[0]<l1: return finder(l0/2,rt[0])
            return finder(l1,l1*2)
        
        labda[i,j], c[i,j] = finder(7,10)
        

plt.figure()
plt.imshow(labda[::-1,:], extent=[q[0],q[-1], P[0],P[-1]])
plt.contour(q,P,labda, 20, colors='k')
plt.xlabel('q (-)')
plt.ylabel('P (-)')
plt.show()

from scipy.optimize import curve_fit

# def l(M,a,b,c,d,e):
#     x,y = M
#     return a*x**(2)*y**c+ d*y**e + b

def l(M,a,b,c,d,e):
    x,y = M
    return a*x**b*y**c+ d*y**e

xdata = np.zeros((2,Np*Nq))
temp=0
for i in range(Np):
    for j in range(Nq):
        xdata[0,temp] = P[i]
        xdata[1,temp] = q[j]
        temp+=1

ydata = []
for i in range(Np):
    for j in range(Nq):
        ydata.append(labda[i,j])

popt, pcov = curve_fit(l, xdata, np.array(ydata), p0=[0,0,0,0,0])


print('a =', popt[0], '+/-', np.sqrt(pcov[0,0]))
print('b =', popt[1], '+/-', np.sqrt(pcov[1,1]))
print('c =', popt[2], '+/-', np.sqrt(pcov[2,2]))
print('d =', popt[3], '+/-', np.sqrt(pcov[3,3]))
print('e =', popt[4], '+/-', np.sqrt(pcov[4,4]))
#print('f =', popt[5], '+/-', np.sqrt(pcov[5,5]))

def _l(P,q,*args):
    return l((P,q), *args)

plt.figure()
c=0
for i in range(0,Np,Np//5):
    plt.plot(q, labda[i,:], '.', color='C'+str(c), label='P = ' + str(round(P[i],3)))
    plt.plot(q, _l(P[i],q, *popt), color='C'+str(c))
    c+=1
plt.xlabel('q')
plt.ylabel('$\Lambda$')
plt.legend()
plt.show()

plt.figure()
c=0
for j in range(0,Nq, Nq//5):
    plt.plot(P,labda[:,j], '.', color = 'C' +str(c), label='q = ' +str(round(q[j],3)))
    plt.plot(P, _l(P,q[j], *popt), color = 'C' +str(c))
    c+=1
plt.xlabel('P')
plt.ylabel('$\Lambda$')
plt.legend()
plt.show()



fig, axes = plt.subplots(1,3, figsize=(16,4))
fig.suptitle(str(q[0]) + ' < q < ' + str(q[-1]))

ax = axes[0]
im = ax.imshow(labda[::-1,:], extent=[q[0],q[-1], P[0],P[-1]], aspect='auto')
ax.contour(q,P,labda, 20, colors='k', inline=True)
ax.set_xlabel('q')
ax.set_ylabel('P')
plt.colorbar(im, ax=ax)

ax = axes[1]
c=0
for i in range(0,Np,Np//5):
    ax.plot(q, labda[i,:], '.', color='C'+str(c), label='P = ' + str(round(P[i],3)))
    ax.plot(q, _l(P[i],q, *popt), color='C'+str(c))
    c+=1
ax.set_xlabel('q')
ax.set_ylabel('$\Lambda$')
ax.legend()

ax = axes[2]
c=0
for j in range(0,Nq, Nq//5):
    ax.plot(P,labda[:,j], '.', color = 'C' +str(c), label='q = ' +str(round(q[j],3)))
    ax.plot(P, _l(P,q[j], *popt), color = 'C' +str(c))
    c+=1
ax.set_xlabel('P')
ax.set_ylabel('$\Lambda$')
ax.legend()

#plt.savefig('save_dir/figures/qlarge')

plt.show()