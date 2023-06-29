#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:09:19 2023

@author: hugo
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi



# Introduce constants, boundary/inital conditions and other parameters
Q = 150  # m/s^3
A = 2000 # m^2
k = 400
kappa = k
R = 10000
L = 10000

D = 2

a = A/(pi*D)


pwr = Q/(kappa*pi*D)

d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
c1 = d1*a**pwr

d2 = 1-d1*R**pwr
c2 = -np.exp(-Q*L/(k*A))*c1

def ssx(x):
    return c1 * np.exp(-Q*x/(k*A)) + c2

def ssr(r):
    return d1*r**pwr + d2

x = np.linspace(0,L,100)
r = np.linspace(a,R,100)

axs = np.append(-x[::-1], r)
res = np.append(ssx(x)[::-1], ssr(r))

plt.figure()
plt.plot(axs, res)
plt.title('steady-state')
plt.xlabel('-x (m) & r (m)')
plt.ylabel('s')
plt.xlim(-L,R)
plt.ylim(0,1)
plt.show()