#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:30:33 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib.animation import FuncAnimation
from numpy import pi



# Introduce constants, boundary/inital conditions and other parameters
Q = 150  # m/s^3
A = 2000 # m^2
k = 400
kappa = k
R = 10000
L = 10000

a = A/pi


from scipy.special import jv
from scipy.special import yv

order = (Q/pi)/(2*kappa)


def phi_bessel(r,par):
    return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))

good_roots_bessel = np.array([[3.94030436e-05,7.61848195e-01], #n=0
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

N_r = 15 #len(good_roots_bessel)
N_x = 15

def phi_r_n(r,n):
    return phi_bessel(r, good_roots_bessel[n-1])

def labda_r_n(n):
    return good_roots_bessel[n-1][0]

def phi_x_n(x,n):
    return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)

def labda_x_n(n):
    return (Q/A)**2/(4*k) + k*(n*pi/L)**2

def ss_x(x):
    s0 = (a/R)**(2*Q/(kappa*pi))
    v = -2*Q/(A*k)
    return s0/(1-np.exp(v*L))*np.exp(v*x) + s0 - s0/(1-np.exp(v*L))


def ss_r(r):
    #return (1-(a/R)**(0.5*Q/(kappa*pi)))/(R-a)*r + 1-R*(1-(a/R)**(0.5*Q/(kappa*pi)))/(R-a)
    return (r/R)**(2*Q/(kappa*pi))

def ssn_x(x):
    s0 = (a/R)**(Q/(kappa*pi))
    v = -Q/(A*k)
    return s0/(1-np.exp(v*L))*np.exp(v*x) + s0 - s0/(1-np.exp(v*L))

def ssn_r(r):
    #return (1-(a/R)**(0.5*Q/(kappa*pi)))/(R-a)*r + 1-R*(1-(a/R)**(0.5*Q/(kappa*pi)))/(R-a)
    return (r/R)**(Q/(kappa*pi))

dt = 1e1
t = np.arange(0,1e6,dt)

# Guess
time_constant = 1.4533828021114656e-05 #min(labda_r_n(1), labda_x_n(1))
#time_constant = labda_r_n(1)
# time_constant = 1
def f_interp(T):
    eq1 = ss_x(0)
    eq2 = ssn_x(0)
    return eq2 +(eq1-eq2)*np.exp(-time_constant*T)

def f_prime_interp(T):
    eq1 = ss_x(0)
    eq2 = ssn_x(0)
    return -time_constant*(eq1-eq2)*np.exp(-time_constant*T)


def psi(r,t):
    return (f_interp(t)-1)/(a-R)*r + (R*f_interp(t)-a)/(R-a)

def xi(x,t):
    return f_interp(t)*(1-x/L)


def u(x):
    return ss_x(x) - xi(x,0)

def v(r):
    return ss_r(r) - psi(r,0)

def H(r,t):
    psi_t = f_prime_interp(t)*r/(a-R) + R*f_prime_interp(t)/(R-a)
    psi_rr = 0
    psi_r = (f_interp(t)-1)/(a-R)
    return -psi_t + kappa*psi_rr + (kappa-Q/pi)/r*psi_r

def Lf(x,t):
    xi_t = f_prime_interp(t)*(1-x/L)
    xi_xx = 0
    xi_x = -f_interp(t)/L
    return -xi_t + k*xi_xx + Q/A*xi_x


def h(tau):
    return np.array([integrate.quad(lambda r: H(r,tau)*phi_r_n(r,n), a, R)[0] for n in range(1,N_r)])

def l(tau):
    return np.array([integrate.quad(lambda x: Lf(x,tau)*phi_x_n(x,n), 0, L)[0] for n in range(1,N_x)])

def inner_x(n,m):
    return integrate.quad(lambda x: phi_x_n(x,m)*phi_x_n(x,n), 0, L)[0]

def inner_r(n,m):
    return integrate.quad(lambda r: phi_r_n(r,m)*phi_r_n(r,n), a, R)[0]

G_x = np.array([[inner_x(n,m) for n in range(1,N_x)] for m in range(1,N_x)])
G_r = np.array([[inner_r(n,m) for n in range(1,N_r)] for m in range(1,N_r)])

labda_x = np.array([labda_x_n(n) for n in range(1,N_x)])
labda_r = np.array([labda_r_n(n) for n in range(1,N_r)])

inv_x = np.linalg.inv(G_x)
inv_r = np.linalg.inv(G_r)


T0_r = inv_r @ np.array([integrate.quad(lambda r: v(r)*phi_r_n(r,n), a, R)[0] for n in range(1,N_r)])
T0_x = inv_x @ np.array([integrate.quad(lambda x: u(x)*phi_x_n(x,n), 0, L)[0] for n in range(1,N_x)])
# def T_x(t):
#     return np.exp(-labda_x*t)*T0_x + np.exp(-labda_x*t) * np.array([integrate.quad(lambda tau: (inv_x @ l(tau)*np.exp(labda_x*tau))[i], 0, t)[0] for i in range(0,N_x-1)])

# def T_r(t):
#     return np.exp(-labda_r*t)*T0_r + np.exp(-labda_r*t) * np.array([integrate.quad(lambda tau: (inv_r @ h(tau)*np.exp(labda_r*tau))[i], 0, t)[0] for i in range(0,N_r-1)])


def T_x(t):
    return np.exp(-labda_x*t)*T0_x + np.exp(-labda_x*t) * integrate.quad_vec(lambda tau: (inv_x @ l(tau) )*np.exp(labda_x*tau), 0, t)[0]

def T_r(t):
    return np.exp(-labda_r*t)*T0_r + np.exp(-labda_r*t) * integrate.quad_vec(lambda tau: (inv_r @ h(tau) )*np.exp(labda_r*tau), 0, t)[0]


def sol_r(r,t):
    res = T_r(t) @ np.array([phi_r_n(r,n) for n in range(1,N_r)])
    return res + psi(r,t)

def sol_x(x,t):
    res = T_x(t) @ np.array([phi_x_n(x,n) for n in range(1,N_x)])
    return res + xi(x,t)

x = np.linspace(0,L,50)
r = np.linspace(a,R,50)

axs = np.append(-x[::-1], r)
res = np.append(sol_x(x,t[0])[::-1], sol_r(r,t[0]))
s0 = np.append(ss_x(x)[::-1], ss_r(r))
sn = np.append(ssn_x(x)[::-1], ssn_r(r))

t = np.linspace(0,1e2,100)

result = [np.zeros(10) for i in range(len(t))]

for i in range(len(t)):
    T = len(t)//10
    if i % T == 0:
        print('time steps @ ' + str(i//T*10) + '%')
    result[i] = np.append(sol_x(x,t[i])[::-1], sol_r(r,t[i]))


def sx(arr):
    n = len(x)
    dx = x[n-1] -x[n-2]
    return (arr[n-1] - arr[n-2])/dx

def sr(arr):
    n = len(x)
    dr = r[1] - r[0]
    return (arr[n+1] - arr[n])/dr

sx_arr = [A*k*sx(result[i]) for i in range(len(result))]
sr_arr = [pi*a*kappa*sr(result[i]) for i in range(len(result))]
ylimit = max(max(sx_arr), max(sr_arr))*1.3


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
    ax.plot(t[:i], sx_arr[:i], label='s_x')
    ax.plot(t[:i], sr_arr[:i], label='s_r')
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, ylimit)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=100, interval=10, repeat=False)
plt.show()


# from matplotlib.animation import PillowWriter
# # #Save the animation as an animated GIF
# # plt.close()
# anim.save("20230509_tc_lr.gif", dpi=80,
#           writer=PillowWriter(fps=10))
