#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:20:36 2023

@author: hugo
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:19:31 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from numpy import pi
from scipy.special import jv
from scipy.special import yv
from scipy.optimize import root
from scipy import integrate

from module_nsol_v3 import fd_sol
from module_nsol_v3 import fd_sol_horde2

Q = 500
A = 7500
k = 900
kappa = 180
L = 45000
R = 7000
D = 20
N_x = 25
N_r = 25
c7 = 2


a = A/(np.pi*D)

m = 101
t = np.linspace(0,5e5,m)

nx, nr, dt = 2000, 6000, 1000
n_steps = int(t[-1]/dt)

dx = L/(nx+1)
dr = (R-a)/(nr+1)

x = np.linspace(0,L,nx+2)[1:-1]
r = np.linspace(a,R,nr+2)[1:-1]
axs = np.append(-x[::-1], r)

def ssx0(x, Q, A, k, kappa, R, L, D, c7):
    pwr = c7*Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-c7*Q*L/(k*A)))
    c1 = d1*a**pwr
    c2 = -np.exp(-c7*Q*L/(k*A))*c1
    return c1 * np.exp(-c7*Q*x/(k*A)) + c2

def ssr0(r, Q, A, k, kappa, R, L, D, c7):
    pwr = c7*Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-c7*Q*L/(k*A)))
    d2 = 1-d1*R**pwr
    return d1*r**pwr + d2

def ssx(x, Q, A, k, kappa, R, L, D):
    a = A/(pi*D)
    pwr = Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
    c1 = d1*a**pwr
    c2 = -np.exp(-Q*L/(k*A))*c1
    return c1 * np.exp(-Q*x/(k*A)) + c2
    
def ssr(r, Q, A, k, kappa, R, L, D):
    a = A/(pi*D)
    pwr = Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
    d2 = 1-d1*R**pwr
    return d1*r**pwr + d2


def timescale(Q, A, k, kappa, R, L, D, nx, nr, dt, n_steps, c7):
    fdres, sx, sr = fd_sol_horde2(Q, A, k, kappa, R, L, D, nx, nr, n_steps, dt, c7)
    t_num = np.arange(0,(n_steps+1)*dt, dt)
    
    x_fd = np.linspace(0,L,nx+2)[1:-1]
    r_fd = np.linspace(a,R,nr+2)[1:-1]
    
    s0_fd = np.append(ssx0(x_fd, Q, A, k, kappa, R, L, D, c7)[::-1], ssr0(r_fd, Q, A, k, kappa, R, L, D, c7))
    sn_fd = np.append(ssx(x_fd, Q, A, k, kappa, R, L, D)[::-1], ssr(r_fd, Q, A, k, kappa, R, L, D))
    
    sx_new = integrate.trapz(ssx(x_fd, Q, A, k, kappa, R, L, D))
    sr_new = integrate.trapz(ssr(r_fd, Q, A, k, kappa, R, L, D)*pi*D*r_fd)
        
    deltas = sn_fd - s0_fd
    deltat = np.zeros(len(axs))
    
    dsx = 0
    dsr = 0
    
    print('Calculating e-folding...\n')
    for i in range(len(axs)):
        for j in range(0,len(t_num)-1, max(1,int((t[1]-t[0])/dt))):
            if abs(sn_fd[i] - fdres[j][i]) <= abs(deltas[i])/20 and deltat[i] == 0:
                deltat[i] = t_num[j]
                
    for j in range(0,len(t_num)-1, max(1,int((t[1]-t[0])/dt))):
            if abs(sx_new - sx[j]) <= abs(sx_new - sx[0])/20 and dsx == 0:
                dsx = t_num[j]
            if abs(sr_new - sr[j]) <= abs(sr_new - sr[0])/20 and dsr == 0:
                dsr = t_num[j]
                print(dsr)
                
    return fdres, deltat, dsx, dsr, sx, sr


def labda0finder(Q, A, k, kappa, R, L, D):
    a = Q/(pi*D)
    order = Q/(2*kappa*pi*D)
    def phi_r(r,par):
        return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))
    
    phi_bca = lambda rt: phi_r(a,rt)
    phi_bcR = lambda rt: phi_r(R,rt)
    
    def func_r(rt):
        return phi_bca(rt),phi_bcR(rt)
    

    def n_roots(rt):
         temp = 0
         if abs(phi_r(a,rt))>0.01: return -1
         elif abs(phi_r(R,rt))>0.01: return -1
         for k in range(10,len(r)-20):
             if phi_r(r[k],rt)*phi_r(r[k-1],rt)<0:
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
    return finder(1e-5,1e-4)


labda_r = labda0finder(Q, A, k, kappa, R, L, D)[0]
labda_x = (Q/A)**2/(4*k) + k*(pi/L)**2
fdr, delta_fd, dsx, dsr, sx, sr = timescale(Q, A, k, kappa, R, L, D, nx, nr, dt, n_steps, c7)
s0 = np.append(ssx0(x, Q, A, k, kappa, R, L, D, c7)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, c7))
sn = np.append(ssx(x, Q, A, k, kappa, R, L, D)[::-1], ssr(r, Q, A, k, kappa, R, L, D))


import matplotlib.patches as mpatches
import matplotlib.lines as mlines

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.plot(axs, delta_fd, '-', color='C0')
ax.hlines(-np.log(0.05)/labda_x, -L,0,ls=':')
ax.hlines(-np.log(0.05)/labda_r, a,R,ls=':')
ax.hlines(dsx, -L,0,ls='--')
ax.hlines(dsr, a,R, ls='--')
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$-x & r$ (m)')
ax.set_ylabel(r'$t$ (s)')
plt.show()


fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.plot(axs/1000, s0, '-', color='C0', label=r'Evenwicht, $Q = 1000$ m$^3$/s')
ax.set_xlim(-L/1000, R/1000)
ax.set_ylim(0,1)
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$s/s_{sea}$')
ax.legend()
plt.savefig("presentatie/intro_evenwicht.pdf")
plt.show()

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.plot(axs/1000, s0, '-', color='C0', label=r'Evenwicht, $Q = 1000$ m$^3$/s')
ax.plot(axs/1000, sn, '-', color='C1', label=r'Evenwicht, $Q = 500$ m$^3$/s')
ax.set_xlim(-L/1000, R/1000)
ax.set_ylim(0,1)
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$s/s_{sea}$')
ax.legend()
plt.savefig("presentatie/intro_evenwicht_nieuw.pdf")
plt.show()



T =  np.arange(0,(n_steps)*dt, dt)

# fig, ax = plt.subplots(1,1, figsize=(8,8))
# for i in range(len(varspace)):
#     ax.plot(T, sx_arr[i])
#     ax.plot(T, sx_arr[i])

# ax.set_xlim(0, t[-1])
# ax.set_xlabel(r'$-x & r$')
# ax.set_ylabel(r'e-folding $t$')
# ax.legend()
# plt.show()

#np.savetxt('save_dir/data/20230522_res_def1_200_200_002.csv', fdres)

##########################
####### ANMINATION #######
##########################

n_skips = int((t[1]-t[0])/dt)
t_num = np.arange(0,(n_steps)*dt, dt)

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.plot(axs/1000, s0, color='C0', label=r'Oud evenwicht, $Q = 1000$ m$^3$/s')
ax.plot(axs/1000, sn, color='C1', label=r'Nieuw evenwicht, $Q = 500$ m$^3$/s')
ax.set_xlim(-L/1000, R/1000)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$s/s_{sea}$')

# Plot the surface.
def init():
    ax.plot(axs/1000, s0, color='C0', label=r'Oud evenwicht, $Q = 1000$ m$^3$/s')
    ax.plot(axs/1000, sn, color='C1', label=r'Nieuw evenwicht, $Q = 500$ m$^3$/s')
    ax.set_xlim(-L/1000, R/1000)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$s/s_{sea}$')
    ax.legend()


# animation function.  This is called sequentially
def animate(i):
    ax.clear()
    ax.plot(axs/1000, s0, color='C0', label=r'Oud evenwicht, $Q = 1000$ m$^3$/s')
    ax.plot(axs/1000, sn, color='C1', label=r'Nieuw evenwicht, $Q = 500$ m$^3$/s')
    ax.plot(axs/1000, fdr[i*n_skips], ls='-', color='k')
    ax.set_xlim(-L/1000, R/1000)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$s/s_{sea}$')
    ax.legend()
    ax.set_title('t=' + str(round(t[i]/3600,1))+'h')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=m-1, interval=10, repeat=False)
plt.show()

from matplotlib.animation import PillowWriter
#Save the animation as an animated GIF
plt.close()
anim.save("presentatie/intro_aanpassing_normaal.gif", dpi=80,
          writer=PillowWriter(fps=10))



####### ANDERE ZEE
R = 70000
x = np.linspace(0,L,nx+2)[1:-1]
r = np.linspace(a,R,nr+2)[1:-1]
axs = np.append(-x[::-1], r)

fdr, delta_fd, dsx, dsr, sx, sr = timescale(Q, A, k, kappa, R, L, D, nx, nr, dt, n_steps, c7)
s0 = np.append(ssx0(x, Q, A, k, kappa, R, L, D, c7)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, c7))
sn = np.append(ssx(x, Q, A, k, kappa, R, L, D)[::-1], ssr(r, Q, A, k, kappa, R, L, D))

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.plot(axs/1000, s0, color='C0', label=r'Oud evenwicht, $Q = 1000$ m$^3$/s')
ax.plot(axs/1000, sn, color='C1', label=r'Nieuw evenwicht, $Q = 500$ m$^3$/s')
ax.set_xlim(-L/1000, 7)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$s/s_{sea}$')

# Plot the surface.
def init():
    ax.plot(axs/1000, s0, color='C0', label=r'Oud evenwicht, $Q = 1000$ m$^3$/s')
    ax.plot(axs/1000, sn, color='C1', label=r'Nieuw evenwicht, $Q = 500$ m$^3$/s')
    ax.set_xlim(-L/1000, 7)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$s/s_{sea}$')
    ax.legend()


# animation function.  This is called sequentially
def animate(i):
    ax.clear()
    ax.plot(axs/1000, s0, color='C0', label=r'Oud evenwicht, $Q = 1000$ m$^3$/s')
    ax.plot(axs/1000, sn, color='C1', label=r'Nieuw evenwicht, $Q = 500$ m$^3$/s')
    ax.plot(axs/1000, fdr[i*n_skips], ls='-', color='k')
    ax.set_xlim(-L/1000, 7)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$s/s_{sea}$')
    ax.legend()
    ax.set_title('t=' + str(round(t[i]/3600,1))+'h')
    ax.legend()
    return 

anim = FuncAnimation(fig, animate, frames=m-1, interval=10, repeat=False)
plt.show()

from matplotlib.animation import PillowWriter
#Save the animation as an animated GIF
plt.close()
anim.save("presentatie/intro_aanpassing_langzaam.gif", dpi=80,
          writer=PillowWriter(fps=10))

