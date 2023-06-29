#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:31:34 2023

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

Q  = 272

k = 900
kappa = 180

L = 45000
R = 7000
A = 7600
D = 20
a = A/(pi*D)

N = 1

k, kappa, L = 3000, 250, 45000

#k, kappa, L = 700, 250, 100000


order = Q/(2*kappa*pi*D)
def phi_r(r,par):
    return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))

  
r = np.linspace(a,R,500)

def good_root_finder(N):
    phi_bc0 = lambda rt: phi_r(a,rt)
    phi_bc1 = lambda rt: phi_r(R,rt)
         
    def func_r(rt):
        return phi_bc0(rt),phi_bc1(rt)
        
    def n_roots(rt):
        temp = 0
        if abs(phi_r(a,rt))>0.001: return -1
        elif abs(phi_r(R,rt))>0.001: return -1
        for k in range(5,len(r)-5):
            if phi_r(r[k],rt)*phi_r(r[k-1],rt)<0:
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

# plt.figure()
# plt.plot(r, phi_r(r, good_root_finder(1)[0]))
# plt.show()

labda_r_1 = good_root_finder(1)[0][0]
labda_x_1 = (Q/A)**2/(4*k) + k*(pi/L)**2

print('sea:', (1/labda_r_1)/(24*3600))
print('river:', (1/labda_x_1)/(24*3600))