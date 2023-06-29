#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:38:29 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(-100,100)

Q0 = 500
Q1 = 1000

plt.figure(figsize=(12,2))
plt.hlines(Q0, t[0], 0, colors = 'k')
plt.hlines(Q1, 0, t[-1], colors = 'k')
plt.vlines(0, Q0, Q1, colors = 'k')
plt.xlabel(r'$t$ (h)')
plt.ylabel(r'$Q$ (m$^3$/s)')
plt.xlim(t[0], t[-1])
plt.ylim(0,1200)
plt.tight_layout()
plt.savefig("presentatie/afvoer_halvering.pdf")
plt.show()

t = np.linspace(-100,100)

Q0 = 250
Q1 = 500

plt.figure(figsize=(12,2))
plt.hlines(Q0, t[0], 0, colors = 'k')
plt.hlines(Q1, 0, t[-1], colors = 'k')
plt.vlines(0, Q0, Q1, colors = 'k')
plt.xlabel(r'$t$ (h)')
plt.ylabel(r'$Q$ (m$^3$/s)')
plt.xlim(t[0], t[-1])
plt.ylim(0,1200)
plt.tight_layout()
plt.savefig("presentatie/afvoer_verdubbeling.pdf")
plt.show()