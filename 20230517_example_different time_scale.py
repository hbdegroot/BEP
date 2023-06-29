#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:58:30 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0,10, 100)

def f(t):
    return -1.8*np.exp(-2*t)

def g(t):
    return 2*np.exp(-t)

y = f(t) + g(t)

T = 0
for j in range(len(t)):
    if y[j] < 0.2/np.e:
        T = t[j]
        break
print(T)


plt.figure()
plt.plot(t, g(t), label=r'$y_1(t) = -2 e^{-t}$')
plt.plot(t, f(t), label=r'$y_2(t) = -1.8 e^{-2t}$')
plt.plot(t, y, label=r'$y_1(t) + y_2(t)$')
plt.axvline(T, color='C2')
plt.axvline(1, color = 'C0')
plt.axvline(0.5, color = 'C1')

plt.xlim(0,5)
plt.xlabel('$t$')
plt.legend()
plt.savefig('exponents.pdf')
plt.show()