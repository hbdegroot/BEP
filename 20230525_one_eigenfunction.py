#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:25:03 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import pi

from scipy.special import jv
from scipy.special import yv
from scipy import integrate

Q, A, k, kappa, R, L, D = 150, 2000, 400, 400, 10000, 10000, 1

a = A/(pi*D)

x = np.linspace(0,L,100)
r = np.linspace(a,R,100)
t = np.linspace(0,4e5,401)

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


order = Q/(2*kappa*pi*D)

# good_roots = np.array([[ 3.25791768e-05,4.06467532e+00,-4.72749879e-01,1.94375538e+00],
# [5.61441414e-05,-1.57234660e+00,2.04451650e-01,8.29972159e-01],
# [1.45935924e-04,1.58585906e+01,-3.63076969e+00,5.18833503e+00],
# [2.51315968e-04,1.85613749e-02,3.83349974e-02,1.10106552e+00],
# [3.54633002e-04,-5.46681567e+00,1.93997243e+00,-3.04898173e-01],
# [5.56400589e-04,1.24640387e+00,-3.78018265e-01, 1.30828925e+00],
# [ 6.94814375e-04 ,-1.18871587e+00 , 7.13740976e-01 , 7.78655817e-01],
# [ 9.39714773e-04 , 3.50340599e+00, -1.56150240e+00 , 1.51968105e+00],
# [ 0.00118613 ,-0.0814833 ,  0.35194703 , 0.9654707 ],
# [ 1.42259592e-03, -2.15540814e+01,  1.36983283e+01 ,-3.06408083e-01],
# [ 0.00178115 , 0.74229588, -0.05655947 , 0.95510943],
# [ 0.0020476 , -1.49598846 , 1.63268582 , 1.05596767],
# [ 0.00244525 , 1.65553452, -0.7369691 ,  0.75081298],
# [ 0.00283119, -0.11893581  ,0.83301684 , 1.02428133],
# [ 3.20578297e-03  ,3.79518454e+00, -2.91930128e+00 ,-7.57465604e-02],
# [0.0037177,  0.60415853, 0.30802913, 0.83396601],
# [ 4.11441077e-03 ,-4.45674408e+00  ,6.58559297e+00 , 3.33354001e+00],
# [ 0.00466465 , 1.11641186 ,-0.25778023 , 0.50024258],
# [ 0.00518971 ,-0.17749531 , 1.83032332 , 1.46349619],
# [ 0.0057034  , 1.56281858 ,-1.11522247 ,-0.07769945],
# [0.00636811, 0.6321072 , 0.79797185, 0.88671047],
# [ 0.00689541 , 2.40228517 ,-3.95352928, -1.895511  ],
# [0.00759819 ,0.94470804, 0.13782939, 0.40899341],
# [ 8.26265996e-03, -6.36387596e-01 , 8.70601038e+00,  5.59895362e+00],
# [ 0.00891516 , 1.01781996 ,-0.47099333, -0.07838152],
# [0.00973298 ,0.90661988 ,1.77422738 ,1.25835381],
# [ 0.01039077 , 0.87552299 ,-1.32570698 ,-0.72769265],
# [0.01124588 ,0.97344635, 0.58487447, 0.40249333],
# [ 0.01205043 , 0.21399645 ,-3.52343512 ,-2.16231823],
# [ 0.01284094 , 0.83257084 ,-0.04139508, -0.08564828]]) #n=16

# good_roots = np.array([[ 2.11361167e-05  ,9.93918054e-01 ,-2.03907245e-01 , 1.70398600e+00],
# [ 4.17398088e-05 ,-4.85507622e+00 , 9.20006757e-01 ,-2.44451389e-01],
# [ 1.12702639e-04 , 9.58970265e-01, -3.16237581e-01 , 1.70543037e+00],
# [ 1.65868791e-04 ,-2.77884855e+00  ,1.09136059e+00 , 2.44356333e-01],
# [ 2.84128647e-04 , 8.85888331e-01, -3.30933495e-01 , 1.54089605e+00],
# [ 3.76027234e-04 ,-2.31582449e+00 , 1.39820693e+00  ,5.31258627e-01],
# [ 5.37517029e-04 , 8.46164531e-01,-3.01797356e-01  ,1.35367772e+00],
# [ 6.71698080e-04, -2.24051561e+00 , 1.84612560e+00  ,8.38842809e-01],
# [ 8.73678771e-04,  8.25304502e-01 ,-2.50664650e-01 , 1.17557242e+00],
# [ 1.05256539e-03 ,-2.41744226e+00 , 2.55500798e+00 , 1.27875907e+00],
# [ 0.00129305 , 0.81364267 ,-0.18738312  ,1.01381536],
# [ 1.51838521e-03, -2.97078598e+00,  3.88730054e+00 , 2.08123673e+00],
# [ 0.00179594  ,0.80641911 ,-0.11657552 , 0.86883933],
# [ 2.06894248e-03 ,-4.70920697e+00 , 7.48041077e+00 , 4.23419711e+00],
# [ 0.00238258  ,0.80152082 ,-0.04030101 , 0.73908825],
# [ 2.70403573e-03, -3.17719426e+01 , 6.07732075e+01 , 3.62052223e+01],
# [0.0030532 , 0.79821048, 0.04078407 ,0.62249402],
# [ 3.42347293e-03,  4.42578267e+00 ,-1.02202613e+01, -6.40120533e+00],
# [0.003808  , 0.7964926 , 0.12690094 ,0.51692899],
# [ 4.22707244e-03 , 1.66531724e+00, -4.71234149e+00, -3.10512355e+00],
# [0.00464715, 0.79680531 ,0.21893121 ,0.42030929],
# [ 0.00511467 , 0.84696382, -3.03449321 ,-2.10730508],
# [0.00557083 ,0.79988268, 0.31837869, 0.3305676 ],
# [ 0.00608611 , 0.45600411 ,-2.20759037 ,-1.61973759],
# [0.00657917, 0.80671895, 0.42746898 ,0.24556473],
# [ 0.00714126  ,0.22622242, -1.70581092 ,-1.32662174],
# [0.0076723  ,0.81861222, 0.54940004 ,0.16295014],
# [ 0.00828004 , 0.07383132 ,-1.36228729, -1.12766302],
# [0.00885031, 0.83729723, 0.68881368, 0.07994715],
# [ 0.00950238 ,-0.03568644, -1.10735418, -0.98099411]]) #n=16

good_roots = np.loadtxt('roots.csv')

def phi_r(r,par):
    return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[2]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))
def phi_x(x,par):
    return np.exp(-Q/(2*k*A)) * (par[1]*np.sin(np.sqrt(4*k*par[0] -(Q/A)**2)/(2*k)* x ) + par[3]*np.cos(np.sqrt(4*k*par[0] -(Q/A)**2)/(2*k)* x ))

def phi_r_n(r,n):
    return phi_r(r, good_roots[n])

def labda_n(n):
    return good_roots[n][0]

def phi_x_n(x,n):
    return phi_x(x,good_roots[n])

def inner(n,m):
    return integrate.quad(lambda x: phi_x_n(x,m)*phi_x_n(x,n), 0, L)[0] + integrate.quad(lambda r: phi_r_n(r,m)*phi_r_n(r,n), a, R)[0]

gamma = 1/(kappa*L/k + R-a)
alpha = -kappa/k*gamma
def psi_x(x):
    return alpha*(x-L)
def psi_r(r):
    return gamma*(r-R) + 1

N = 20
G = np.array([[inner(n,m) for n in range(N)] for m in range(N)])


labda= np.array([labda_n(n) for n in range(N)])

inv = np.linalg.inv(G)

I1 = [sum([inv[n,j]*(integrate.quad(lambda x: (ssx0(x)-psi_x(x))*phi_x_n(x,j), 0, L)[0] + integrate.quad(lambda r: (ssr0(r)-psi_r(r))*phi_r_n(r,j), a, R)[0]) for j in range(N)]) for n in range(N)]
I2 = [sum([inv[n,j]*(integrate.quad(lambda x: alpha*Q/A*phi_x_n(x,j), 0, L)[0] + integrate.quad(lambda r: gamma*(kappa-Q/(pi*D))/r*phi_r_n(r,j), a, R)[0]) for j in range(N)]) for n in range(N)]

tempix1 = np.array([I1[n] -I2[n]/labda[n] for n in range(N)])
tempix2 = np.array([I2[n]/labda[n] for n in range(N)])

tempitx1 = np.array([np.exp(-labda[n]*t) for n in range(N)])

Tx = np.zeros((len(t), N))

for i in range(len(t)):
    T1 = tempitx1[:,i] * tempix1
    T2 = tempix2
    Tx[i,:] = T1 + T2
    
axs = np.append(-x[::-1], r)

PSI = np.zeros((len(t), len(axs)))
for i in range(len(t)):
    for j in range(len(x)):
        PSI[i,j] = psi_x(-axs[j])
    for j in range(len(r)):
        PSI[i,j+len(x)] = psi_r(r[j])

PHI = np.array([np.append(phi_x_n(x,n)[::-1],phi_r_n(r,n)) for n in range(N)]) 
    
solx = Tx @ PHI + PSI
  

s0 = np.append(ssx0(x)[::-1], ssr0(r))
sn = np.append(ssx(x)[::-1], ssr(r))

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
# ax.set_ylim(0, ylimit)
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
    # ax.set_ylim(0, ylimit)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    return 

# animation function.  This is called sequentially
def animate(i):
    ax = axes[0]
    ax.clear()
    ax.plot(axs, s0, label='initial')
    ax.plot(axs, sn, label='final')
    ax.plot(axs, solx[i], label = 'eigenfunction expansion')
    ax.plot(axs[len(x)], solx[i,len(x)], 'r.')
    ax.set_xlim(-L, R)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$s$')
    ax.set_title('t=' + str(t[i]))
    ax.legend()
    
    ax = axes[1]
    ax.clear()
    ax.set_xlim(0, t[-1])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$ds/dx & ds/dr$')
    # ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=100-1, interval=10, repeat=False)
plt.show()


deltat = 0 
for j in range(len(t)):
    if (ssx(0) - solx[j,len(x)]) < (ssx(0) - ssx0(0))/np.e and deltat == 0:
        deltat = t[j]
        
print(deltat)

deltat2 = 0 
for j in range(len(t)):
    if (solx[len(t)-1,len(x)] - solx[j,len(x)]) < (solx[len(t)-1,len(x)]- solx[0,len(x)])/np.e and deltat2 == 0:
        deltat2 = t[j]
        
print(deltat2)


plt.figure()
plt.plot(t, solx[:,len(x)])
plt.axvline(deltat)
plt.axvline(deltat2)
plt.show()


