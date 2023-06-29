#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 20:28:22 2023

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


Q, A, k, kappa, L, R, D, N = 500, 7500, 900, 180, 45000, 7000, 20, 10

a = A/(pi*D)

x = np.linspace(0,L,1000)
r = np.linspace(a,R,1000)


def phi_x_n(x,n):
    n+=1
    return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)


def labda_x_n(n):
    n+=1
    return (Q/A)**2/(4*k) + k*(n*pi/L)**2


print('Looking for some roots...')
good_roots = goodrootfinder(N, 1e-5, 5e-3, 50, Q, A, kappa, R, 0, D)
print('All set, continue integrating \n')

order = Q/(2*kappa*pi*D)
def phi_r(r,par):
    return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))

sign = [-np.sign(phi_r(0.01, good_roots[n]))/np.sqrt(integrate.quad(lambda r: phi_r(r,good_roots[n])**2, a, R)[0])*np.sqrt(R-a) for n in range(N)]

def phi_r_n(r,n):
    return phi_r(r, good_roots[n]) * sign[n]

def labda_r_n(n):
    return good_roots[n][0]


plt.figure(figsize=(5,5))
for n in range(5):
    plt.plot(-x, phi_x_n(x,n), label=r'$n = $' + str(n+1))
# plt.xlabel(r'$x$')
# plt.ylabel(r'$\phi_n(x)$')
plt.xlim(-L,0)
plt.ylim(-1,1)
plt.legend()
plt.xticks([])
plt.yticks([])
plt.title('Eigenfuncties Rivier')
plt.xlabel('$x$ (km)')
plt.savefig("presentatie/eigenfunctions_river.pdf")
plt.show()

plt.figure(figsize=(5,5))
for n in range(5):
    plt.plot(r, phi_r_n(r,n), label=r'$n = $' + str(n+1))
# plt.xlabel(r'$r$')
# plt.ylabel(r'$\phi_n(r)$')
plt.xlim(a,R)
plt.ylim(-3,3)
plt.legend()
plt.xticks([])
plt.yticks([])
plt.title('Eigenfuncties Zee')
plt.xlabel('$r$ (km)')
plt.savefig("presentatie/eigenfunctions_sea.pdf")
plt.show()



plt.figure(figsize=(8,8))
for n in range(3):
    plt.plot(-x/1000, phi_x_n(x,n), label=r'$n = $' + str(n+1))
plt.xlabel(r'$-x$ (km)')
plt.ylabel(r'$\phi_n(x)$')
plt.xlim(-L/1000,0)
plt.ylim(-1,1)
plt.legend()
# plt.xticks([])
# plt.yticks([])
plt.title('Eigenfuncties Rivier')
plt.xlabel('$x$ (km)')
plt.savefig("presentatie/eigenfunctions_signs_river.pdf")
plt.show()

plt.figure(figsize=(8,8))
for n in range(2):
    plt.plot(r/1000, phi_r_n(r,n), label=r'$n = $' + str(n+1))
plt.xlabel(r'$r$ (km)')
plt.ylabel(r'$\phi_n(r)$')
plt.xlim(a/1000,R/1000)
plt.ylim(-3,3)
plt.legend()
# plt.xticks([])
# plt.yticks([])
plt.title('Eigenfuncties Zee')
plt.xlabel('$r$ (km)')
plt.savefig("presentatie/eigenfunctions_signs_zee.pdf")
plt.show()