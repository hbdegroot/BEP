#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 16:57:31 2023

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
kappa = k
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

N_r = 18
N_x = 18

#We now count from n=0 for simplicity and to add confusion :)

def phi_r_n(r,n):
    return phi_r(r, good_roots_r[n])

def labda_r_n(n):
    return good_roots_r[n][0]

def phi_r_prime(r,n):
    return phi_r_prime0(r, good_roots_r[n])

def phi_r_dprime(r,n):
    return phi_r_dprime0(r, good_roots_r[n])


def phi_x_n(x,n):
    n+=1
    return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)

def labda_x_n(n):
    n+=1
    return (Q/A)**2/(4*k) + k*(n*pi/L)**2

def phi_x_prime(x,n):
    n+=1
    return n*pi/L

def phi_x_dprime(x,n):
    n+=1
    return -2*n*pi/L*(-Q)/(2*k*A)

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


alpha = ssx(0)
beta = ssx0(0)-ssx(0)
gamma = 1.4533828021114656e-05
gamma = 2.92478726253855e-05
def f(t):
    return alpha + beta * np.exp(-gamma*t)

f0 = ssx0(0)


I1 = [sum([inv_x[n,j]*integrate.quad(lambda x: ssx0(x)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
I2 = [sum([inv_x[n,j]*integrate.quad(lambda x: (1-x/L)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
I3 = [sum([inv_x[n,j]*integrate.quad(lambda x: (-Q/(A*L))*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]

P0 = [sum([inv_r[n,j]*integrate.quad(lambda r: (r-a)/(R-a)*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
P1 = [sum([inv_r[n,j]*integrate.quad(lambda r: ssr0(r)*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
P2 = [sum([inv_r[n,j]*integrate.quad(lambda r: (R-r)/(R-a)*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
P3 = [sum([inv_r[n,j]*integrate.quad(lambda r: 1/(a-R)*(kappa-Q/(pi*D))/r*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]

tempix1 = np.array([I1[n] - f0*I2[n] for n in range(N_x)])
tempix2 = np.array([-I2[n] for n in range(N_x)])
tempix3 = np.array([I3[n] for n in range(N_x)])

tempir0 = np.array([-P3[n] for n in range(N_r)])
tempir1 = np.array([-P0[n] + P1[n] - f0*P2[n] + P3[n] for n in range(N_r)])
tempir2 = np.array([-P2[n] for n in range(N_r)])
tempir3 = np.array([P3[n] for n in range(N_r)])


m = 100
t = np.linspace(0,1e5,m)
x = np.linspace(0,L,100)
r = np.linspace(a,R,100)

tempitx1 = np.array([np.exp(-labda_x[n]*t) for n in range(N_x)])
tempitx2 = np.array([-beta*gamma/(labda_x[n]-gamma)*(np.exp(-gamma*t) - np.exp(-labda_x[n]*t)) for n in range(N_x)])
tempitx3 = np.array([alpha/labda_x[n]*(1-np.exp(-labda_x[n]*t)) + beta/(labda_x[n]-gamma)*(np.exp(-gamma*t) - np.exp(-labda_x[n]*t)) for n in range(N_x)])

tempitr0 = np.array([1/labda_r[n]*(1-np.exp(-labda_r[n]*t)) for n in range(N_r)])
tempitr1 = np.array([np.exp(-labda_r[n]*t) for n in range(N_r)])
tempitr2 = np.array([-beta*gamma/(labda_r[n]-gamma)*(np.exp(-gamma*t) - np.exp(-labda_r[n]*t)) for n in range(N_r)])
tempitr3 = np.array([alpha/labda_r[n]*(1-np.exp(-labda_r[n]*t)) + beta/(labda_r[n]-gamma)*(np.exp(-gamma*t) - np.exp(-labda_r[n]*t)) for n in range(N_r)])

Tx = np.zeros((len(t), N_x))
Tr = np.zeros((len(t), N_r))

for i in range(len(t)):
    T1 = tempitx1[:,i] * tempix1
    T2 = tempitx2[:,i] * tempix2
    T3 = tempitx3[:,i] * tempix3
    Tx[i,:] = T1 + T2 + T3
    
for i in range(len(t)):
    T0 = tempitr0[:,i] * tempir0
    T1 = tempitr1[:,i] * tempir1
    T2 = tempitr2[:,i] * tempir2
    T3 = tempitr3[:,i] * tempir3
    Tr[i,:] = T0 + T1 + T2 + T3
    
    
def ksi(x,t):
    return f(t)*(1-x/L)

def psi(r,t):
    return (f(t)-1)*r/(a-R)+(R*f(t)-a)/(R-a)

KSI = np.zeros((len(t), len(x)))
PSI = np.zeros((len(t), len(r)))
for i in range(len(t)):
    for j in range(len(x)):
        KSI[i,j] = ksi(x[j], t[i])

for i in range(len(t)):
    for j in range(len(r)):
        PSI[i,j] = psi(r[j], t[i])
    
solx = Tx @ np.array([phi_x_n(x,n) for n in range(N_x)]) + KSI
solr = Tr @ np.array([phi_r_n(r,n) for n in range(N_r)]) + PSI



axs = np.append(-x[::-1], r)
res = np.append(solx[0][::-1], solr[0])
s0 = np.append(ssx0(x)[::-1], ssr0(r))
sn = np.append(ssx(x)[::-1], ssr(r))


result = [np.zeros(10) for i in range(len(t))]
for i in range(len(t)):
    result[i] = np.append(solx[i][::-1], solr[i])



################ DERIVATIVES
def ksi_dx(x,t):
    return -f(t)/L
def psi_dr(r,t):
    return (f(t)-1)/(a-R)

KSI_dx = np.zeros(len(t))
PSI_dr = np.zeros(len(t))
for i in range(len(t)):
    KSI_dx[i] = ksi_dx(0, t[i])

for i in range(len(t)):
    PSI_dr[i] = psi_dr(a, t[i])
    
dsdx = Tx @ np.array([phi_x_prime(0,n) for n in range(N_x)]) + KSI_dx
dsdr = Tr @ np.array([phi_r_prime(a,n) for n in range(N_r)]) + PSI_dr

ds2dx2 = Tx @ np.array([phi_x_dprime(0,n) for n in range(N_x)])
ds2dr2 = Tr @ np.array([phi_r_dprime(a,n) for n in range(N_r)])

transx = k*ds2dx2 + Q/A*dsdx
transr = kappa*ds2dr2 + 1/a*(kappa-Q/(pi*D))*dsdr

def sx(arr):
    n = len(x)
    dx = x[n-1] -x[n-2]
    dsdx = (arr[n-1] - arr[n-2])/dx
    ds2dx2 = (arr[n-1] -2* arr[n-2] + arr[n-3])/dx**2
    return k*ds2dx2 + Q/A*dsdx

def sr(arr):
    n = len(x)
    dr = r[1] - r[0]
    dsdr = (arr[n+1] - arr[n])/dr
    ds2dr2 = (arr[n+2]-2*arr[n+1] + arr[n])/dr**2
    return kappa*ds2dr2 + 1/a*(kappa-Q/(pi*D))*dsdr

sx_arr = [sx(result[i]) for i in range(len(result))]
sr_arr = [sr(result[i]) for i in range(len(result))]
ylimit = max(max(sx_arr), max(sr_arr))*1.3

############## PLOTTING


fig, axes = plt.subplots(2,1, figsize=(8,8))

ax = axes[0]
ax.plot(axs, s0)
ax.plot(axs, sn)
ax.set_xlim(-L, R)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$-x & r$')
ax.set_ylabel(r'$s$')

ax = axes[1]
ax.set_xlim(0, t[-1])
ax.set_ylim(0, ylimit)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$ds/dx & ds/dr$')

# Plot the surface.
def init():
    ax = axes[0]
    ax.plot(axs, s0, label='initial')
    ax.plot(axs, sn, label='final')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    
    ax = axes[1]
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, ylimit)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax = axes[0]
    ax.clear()
    ax.plot(axs, s0, label='initial')
    ax.plot(axs, sn, label='final')
    ax.plot(axs, result[i], label=str(t[i]))
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.legend()
    
    ax = axes[1]
    ax.clear()
    # ax.plot(t[:i], sx_arr[:i], label='s_x')
    # ax.plot(t[:i], sr_arr[:i], label='s_r')
    # ax.plot(t[:i], -dsdx[:i], 'r--' )
    # ax.plot(t[:i], dsdr[:i], 'k--')
    ax.plot(t[:i], -transx[:i], 'r--', label='transx')
    ax.plot(t[:i], transr[:i], 'k--', label='transr')
    ax.set_xlim(0, t[-1])
    # ax.set_ylim(0, ylimit)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=m, interval=10, repeat=False)
plt.show()


# plt.figure()
# plt.plot(x,ssx0(x), 'k--')
# plt.plot(x,ssx(x), 'r--')
# for i in range(len(t)):
#     plt.plot(x, solx[i])
# plt.show()


# plt.figure()
# plt.plot(r,ssr0(r), 'k--')
# plt.plot(r,ssr(r), 'r--')
# for i in range(len(t)):
#     plt.plot(r, solr[i])
# plt.show()
