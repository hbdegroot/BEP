#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:16:01 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy import integrate
from scipy.special import jv
from scipy.special import yv

Q = 150  # m/s^3
A = 2000 # m^2
k = 400
kappa = k/2
R = 10000
L = 10000

D = 1

pi = np.pi

a = A/(pi*D)

def ssx(x):
    pwr = Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
    c1 = d1*a**pwr
    c2 = -np.exp(-Q*L/(k*A))*c1
    return c1 * np.exp(-Q*x/(k*A)) + c2

def ssr(r):
    pwr = Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
    d2 = 1-d1*R**pwr
    return d1*r**pwr + d2

def ssx0(x):
    pwr = 2*Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-2*Q*L/(k*A)))
    c1 = d1*a**pwr
    c2 = -np.exp(-2*Q*L/(k*A))*c1
    return c1 * np.exp(-2*Q*x/(k*A)) + c2

def ssr0(r):
    pwr = 2*Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-2*Q*L/(k*A)))
    d2 = 1-d1*R**pwr
    return d1*r**pwr + d2


order = Q/(2*kappa*pi)

def phi_r(r,par):
    return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))

def phi_r_prime0(r,par):
    c = order
    w = np.sqrt(par[0]/kappa)
    d = par[1]
    
    T1 = d*w*r*yv(c-1, w*r)
    T2 = 2*c*d*yv(c, w*r)
    T3 = -d*w*r*yv(c+1,w*r)
    T4 = w*r*jv(c-1,w*r)
    T5 = 2*c*jv(c,w*r)
    T6 = -w*r*jv(c+1,w*r)
    return 0.5*r**(c-1)*(T1+T2+T3+T4+T5+T6)

def phi_r_dprime0(r,par):
    c = order
    w = np.sqrt(par[0]/kappa)
    d = par[1]
    
    T11 = (c**2+c-w**2*r**2)*yv(c, w*r)/(w**2*r**2)
    T12 = yv(c-1,w*r)/(w*r)
    T1 = d*w**2*r**c*(T11 - T12)
    
    T21 = (c**2+c-w**2*r**2)*jv(c, w*r)/(w**2*r**2)
    T22 = jv(c-1,w*r)/(w*r)
    T2 = w**2*r**c*(T21-T22)
    
    T3 = (c-1)*c*d*r**(c-2)*yv(c,w*r) + c*d*w*r**(c-1)*(yv(c-1,w*r) - yv(c+1,w*r))
    T4 = (c-1)*c*r**(c-2)*jv(c,w*r) + c*w*r**(c-1)*(jv(c-1,w*r) - jv(c+1,w*r))
    return (T1+T2+T3+T4)

good_roots_r = np.array([[3.94030436e-05,7.61848195e-01], #n=0
                          [1.71476791e-04,1.36154281e+00], #n=1
                          [3.94842613e-04,2.49685017e+00], #n=2
                          [7.08883157e-04,6.97900847e+00], #n=3
                          [1.11333787e-03,-1.12220582e+01], #n=4
                          [1.60807412e-03,-3.06309775e+00], #n=5
                          [2.19302e-03,-1.67100131e+00], #n=6
                          [2.86812e-03,-1.05017646e+00], #n=7
                          [3.63335e-03,-0.66912985e+00], #n=8
                          [4.4887e-03,-0.38862351e+00], #n=9
                          [5.43415e-03,-0.15366476e+00], #n=10
                          [6.46968e-03,0.065137e+00], #n=11
                          [7.5953e-03,0.2897497e+00], #n=12
                          [8.81099e-03,0.54445193e+00], #n=13
                          [1.011676e-02,0.86784664e+00], #n=14
                          [1.15126e-02,1.34271697e+00], #n=15
                          [1.29985e-02,2.21252377e+00], #n=16
                          [1.457447e-02,4.71344792e+00], #n=17
                          [1.62405098e-02,-2.61426615e+02], #n=18
                          [1.799661e-02,-4.44488786e+00]]) #n=20

N_r = 19
N_x = 19

#We now count from n=0 for simplicity and to add confusion :)

def phi_r_n(r,n):
    if n==0: return ssr(r)
    return phi_r(r, good_roots_r[n-1])

def labda_r_n(n):
    if n==0: return 0
    return good_roots_r[n-1][0]

# def phi_r_prime(r,n):
#     return phi_r_prime0(r, good_roots_r[n])

# def phi_r_dprime(r,n):
#     return phi_r_dprime0(r, good_roots_r[n])


def phi_x_n(x,n):
    if n==0: return ssx(x)
    return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)

def labda_x_n(n):
    if n==0: return 0
    return (Q/A)**2/(4*k) + k*(n*pi/L)**2

# def phi_x_prime(x,n):
#     n+=1
#     return n*pi/L

# def phi_x_dprime(x,n):
#     n+=1
#     return -2*n*pi/L*(-Q)/(2*k*A)

def inner_x(n,m):
    return integrate.quad(lambda x: phi_x_n(x,m)*phi_x_n(x,n), 0, L)[0]

def inner_r(n,m):
    return integrate.quad(lambda r: phi_r_n(r,m)*phi_r_n(r,n), a, R)[0]

G_x = np.array([[inner_x(n,m) for n in range(N_x)] for m in range(N_x)])
G_r = np.array([[inner_r(n,m) for n in range(N_r)] for m in range(N_r)])

labda_x = np.array([labda_x_n(n) for n in range(N_x)])
labda_r = np.array([labda_r_n(n) for n in range(N_r)])

inv_x = np.linalg.inv(G_x)
inv_r = np.linalg.inv(G_r)


I1 = [sum([inv_x[n,j]*integrate.quad(lambda x: ssx0(x)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
I2 = [sum([inv_x[n,j]*integrate.quad(lambda x: A/Q*(x-L)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
I3 = [sum([inv_x[n,j]*integrate.quad(lambda x: phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]

P1 = [sum([inv_r[n,j]*integrate.quad(lambda r: (ssr0(r)-1)*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
P2 = [sum([inv_r[n,j]*integrate.quad(lambda r: -a*(r-R)/(kappa-Q/(pi*D))*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
P3 = [sum([inv_r[n,j]*integrate.quad(lambda r: a/r*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]

alpha = P3[0]*phi_r_n(a,0) - I3[0]*phi_x_n(0,0)
beta = P2[0]*phi_r_n(a,0) + I2[0]*phi_x_n(0,0) + a*(a-R)/(kappa-Q/(pi*D)) + A*L/Q

print(alpha/beta)
