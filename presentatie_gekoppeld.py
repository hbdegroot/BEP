 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 19:59:55 2023

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

m = 1001
t = np.linspace(0,2e6,m)

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

#varspace = np.array([2,0.5])        
varspaceQ = np.array([0.5,2])
varspace = np.array([kappa/10, kappa/4, kappa/2, kappa*2])  

labda_r = []
labda_x = []
delta = []
fd = []
tx, tr = [], []
sx_arr, sr_arr = [], []
sn, s0 = [], []

labda_r2 = []
labda_x2 = []
delta2 = []
fd2 = []
tx2, tr2 = [], []
sx_arr2, sr_arr2 = [], []
sn2, s02 = [], []

for i in range(len(varspace)):
    kappa = varspace[i]
    c7 = varspaceQ[0]
    labda_r.append(labda0finder(Q, A, k, kappa, R, L, D)[0])
    labda_x.append((Q/A)**2/(4*k) + k*(pi/L)**2)
    fdr, deltat_fd, dsx, dsr, sx, sr = timescale(Q, A, k, kappa, R, L, D, nx, nr, dt, n_steps, c7)
    delta.append(deltat_fd)
    tx.append(dsx)
    tr.append(dsr)
    fd.append(fdr)
    sx_arr.append(sx)
    sr_arr.append(sr)
    s0.append(np.append(ssx0(x, Q, A, k, kappa, R, L, D, c7)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, c7)))
    sn.append(np.append(ssx(x, Q, A, k, kappa, R, L, D)[::-1], ssr(r, Q, A, k, kappa, R, L, D)))
    
    c7 = varspaceQ[1]
    labda_r2.append(labda0finder(Q, A, k, kappa, R, L, D)[0])
    labda_x2.append((Q/A)**2/(4*k) + k*(pi/L)**2)
    fdr2, deltat_fd2, dsx2, dsr2, sx2, sr2 = timescale(Q, A, k, kappa, R, L, D, nx, nr, dt, n_steps, c7)
    delta2.append(deltat_fd2)
    tx2.append(dsx2)
    tr2.append(dsr2)
    fd2.append(fdr2)
    sx_arr2.append(sx2)
    sr_arr2.append(sr2)
    s02.append(np.append(ssx0(x, Q, A, k, kappa, R, L, D, c7)[::-1], ssr0(r, Q, A, k, kappa, R, L, D, c7)))
    sn2.append(np.append(ssx(x, Q, A, k, kappa, R, L, D)[::-1], ssr(r, Q, A, k, kappa, R, L, D)))



import matplotlib.patches as mpatches
import matplotlib.lines as mlines

L = L/1000
R = R/1000
a = a/1000
axs = axs/1000

delta = [delta[i]/3600 for i in range(len(delta))]
delta2 = [delta2[i]/3600 for i in range(len(delta2))]
labda_x = [labda_x[i]*3600 for i in range(len(labda_x))]
labda_r = [labda_r[i]*3600 for i in range(len(labda_r))]
labda_x2 = [labda_x2[i]*3600 for i in range(len(labda_x2))]
labda_r2 = [labda_r2[i]*3600 for i in range(len(labda_r2))]
t = t/3600
####   AFNAME


fig, ax = plt.subplots(1,1, figsize=(8,8))
for i in range(3,4):
    ax.plot(axs, delta[i], '-', color='C' + str(i), label='$\kappa$ = '+str(round(varspace[i],1)))
    ax.hlines(-np.log(0.05)/labda_x[i], -L,0,ls='--', colors='C'+str(i))
    ax.hlines(-np.log(0.05)/labda_r[i], a,R,ls='--', colors='C' + str(i))
    # ax.hlines(tx[i], -L,0,ls='--', colors='C' + str(i))
    # ax.hlines(tr[i], a,R, ls='--', colors='C' + str(i))
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$x & r$ (km)')
ax.set_ylabel(r'$t$ (h)')

dt_line = mlines.Line2D([], [], color='k', ls='-', label=r'$t_{ADJ}$')
# dt2_line = mlines.Line2D([], [], color='k', ls='--', label=r'$T_{ADJ}$')
dt3_line = mlines.Line2D([], [], color='k', ls='--', label=r'$1/\lambda_1$')
c_patches = [mpatches.Patch(color='C'+str(j), label=r'$\kappa$ = '+str(round(varspace[j],1)) + ' m$^2$/s' ) for j in range(len(varspace))]
ax.legend(handles=c_patches+[dt_line, dt3_line], loc='upper left')
plt.savefig("presentatie/coupled_decreasing_1.pdf")
plt.show()

fig, ax = plt.subplots(1,1, figsize=(8,8))
for i in range(2,4):
    ax.plot(axs, delta[i], '-', color='C' + str(i), label='$\kappa$ = '+str(round(varspace[i],1)))
    ax.hlines(-np.log(0.05)/labda_x[i], -L,0,ls='--', colors='C'+str(i))
    ax.hlines(-np.log(0.05)/labda_r[i], a,R,ls='--', colors='C' + str(i))
    # ax.hlines(tx[i], -L,0,ls='--', colors='C' + str(i))
    # ax.hlines(tr[i], a,R, ls='--', colors='C' + str(i))
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$x & r$ (km)')
ax.set_ylabel(r'$t$ (h)')

dt_line = mlines.Line2D([], [], color='k', ls='-', label=r'$t_{ADJ}$')
dt3_line = mlines.Line2D([], [], color='k', ls='--', label=r'$1/\lambda_1$')
c_patches = [mpatches.Patch(color='C'+str(j), label=r'$\kappa$ = '+str(round(varspace[j],1)) + ' m$^2$/s' ) for j in range(len(varspace))]
ax.legend(handles=c_patches+[dt_line, dt3_line], loc='upper left')
plt.savefig("presentatie/coupled_decreasing_2.pdf")
plt.show()

fig, ax = plt.subplots(1,1, figsize=(8,8))
for i in range(1,4):
    ax.plot(axs, delta[i], '-', color='C' + str(i), label='$\kappa$ = '+str(round(varspace[i],1)))
    ax.hlines(-np.log(0.05)/labda_x[i], -L,0,ls='--', colors='C'+str(i))
    ax.hlines(-np.log(0.05)/labda_r[i], a,R,ls='--', colors='C' + str(i))
    # ax.hlines(tx[i], -L,0,ls='--', colors='C' + str(i))
    # ax.hlines(tr[i], a,R, ls='--', colors='C' + str(i))
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$x & r$ (km)')
ax.set_ylabel(r'$t$ (h)')

dt_line = mlines.Line2D([], [], color='k', ls='-', label=r'$t_{ADJ}$')
#dt2_line = mlines.Line2D([], [], color='k', ls='--', label=r'$T_{ADJ}$')
dt3_line = mlines.Line2D([], [], color='k', ls='--', label=r'$1/\lambda_1$')
c_patches = [mpatches.Patch(color='C'+str(j), label=r'$\kappa$ = '+str(round(varspace[j],1)) + ' m$^2$/s' ) for j in range(len(varspace))]
ax.legend(handles=c_patches+[dt_line, dt3_line], loc='upper left')
plt.savefig("presentatie/coupled_decreasing_3.pdf")
plt.show()


fig, ax = plt.subplots(1,1, figsize=(8,8))
for i in range(len(varspace)):
    ax.plot(axs, delta[i], '-', color='C' + str(i), label='$\kappa$ = '+str(round(varspace[i],1)))
    ax.hlines(-np.log(0.05)/labda_x[i], -L,0,ls='--', colors='C'+str(i))
    ax.hlines(-np.log(0.05)/labda_r[i], a,R,ls='--', colors='C' + str(i))
    # ax.hlines(tx[i], -L,0,ls='--', colors='C' + str(i))
    # ax.hlines(tr[i], a,R, ls='--', colors='C' + str(i))
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$x & r$ (km)')
ax.set_ylabel(r'$t$ (h)')

dt_line = mlines.Line2D([], [], color='k', ls='-', label=r'$t_{ADJ}$')
# dt2_line = mlines.Line2D([], [], color='k', ls='--', label=r'$T_{ADJ}$')
dt3_line = mlines.Line2D([], [], color='k', ls='--', label=r'$1/\lambda_1$')
c_patches = [mpatches.Patch(color='C'+str(j), label=r'$\kappa$ = '+str(round(varspace[j],1)) + ' m$^2$/s' ) for j in range(len(varspace))]
ax.legend(handles=c_patches+[dt_line, dt3_line], loc='upper left')
plt.savefig("presentatie/coupled_decreasing_4.pdf")
plt.show()



####    TOENAME

fig, ax = plt.subplots(1,1, figsize=(8,8))
for i in range(3,4):
    ax.plot(axs, delta2[i], '-', color='C' + str(i), label='$\kappa$ = '+str(round(varspace[i],1)))
    ax.hlines(-np.log(0.05)/labda_x2[i], -L,0,ls='--', colors='C'+str(i))
    ax.hlines(-np.log(0.05)/labda_r2[i], a,R,ls='--', colors='C' + str(i))
    # ax.hlines(tx2[i], -L,0,ls='--', colors='C' + str(i))
    # ax.hlines(tr2[i], a,R, ls='--', colors='C' + str(i))
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$x & r$ (km)')
ax.set_ylabel(r'$t$ (h)')

dt_line = mlines.Line2D([], [], color='k', ls='-', label=r'$t_{ADJ}$')
# dt2_line = mlines.Line2D([], [], color='k', ls='--', label=r'$T_{ADJ}$')
dt3_line = mlines.Line2D([], [], color='k', ls='--', label=r'$1/\lambda_1$')
c_patches = [mpatches.Patch(color='C'+str(j), label=r'$\kappa$ = '+str(round(varspace[j],1))  + ' m$^2$/s' )for j in range(len(varspace))]
ax.legend(handles=c_patches+[dt_line, dt3_line])
plt.savefig("presentatie/coupled_increasing_1.pdf")
plt.show()

fig, ax = plt.subplots(1,1, figsize=(8,8))
for i in range(2,4):
    ax.plot(axs, delta2[i], '-', color='C' + str(i), label='$\kappa$ = '+str(round(varspace[i],1)))
    ax.hlines(-np.log(0.05)/labda_x2[i], -L,0,ls='--', colors='C'+str(i))
    ax.hlines(-np.log(0.05)/labda_r2[i], a,R,ls='--', colors='C' + str(i))
    # ax.hlines(tx2[i], -L,0,ls='--', colors='C' + str(i))
    # ax.hlines(tr2[i], a,R, ls='--', colors='C' + str(i))
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$x & r$ (km)')
ax.set_ylabel(r'$t$ (h)')

dt_line = mlines.Line2D([], [], color='k', ls='-', label=r'$t_{ADJ}$')
# dt2_line = mlines.Line2D([], [], color='k', ls='--', label=r'$T_{ADJ}$')
dt3_line = mlines.Line2D([], [], color='k', ls='--', label=r'$1/\lambda_1$')
c_patches = [mpatches.Patch(color='C'+str(j), label=r'$\kappa$ = '+str(round(varspace[j],1))  + ' m$^2$/s' )for j in range(len(varspace))]
ax.legend(handles=c_patches+[dt_line, dt3_line])
plt.savefig("presentatie/coupled_increasing_2.pdf")
plt.show()

fig, ax = plt.subplots(1,1, figsize=(8,8))
for i in range(1,4):
    ax.plot(axs, delta2[i], '-', color='C' + str(i), label='$\kappa$ = '+str(round(varspace[i],1)))
    ax.hlines(-np.log(0.05)/labda_x2[i], -L,0,ls='--', colors='C'+str(i))
    ax.hlines(-np.log(0.05)/labda_r2[i], a,R,ls='--', colors='C' + str(i))
    # ax.hlines(tx2[i], -L,0,ls='--', colors='C' + str(i))
    # ax.hlines(tr2[i], a,R, ls='--', colors='C' + str(i))
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$x & r$ (km)')
ax.set_ylabel(r'$t$ (h)')

dt_line = mlines.Line2D([], [], color='k', ls='-', label=r'$t_{ADJ}$')
# dt2_line = mlines.Line2D([], [], color='k', ls='--', label=r'$T_{ADJ}$')
dt3_line = mlines.Line2D([], [], color='k', ls='--', label=r'$1/\lambda_1$')
c_patches = [mpatches.Patch(color='C'+str(j), label=r'$\kappa$ = '+str(round(varspace[j],1))  + ' m$^2$/s' )for j in range(len(varspace))]
ax.legend(handles=c_patches+[dt_line, dt3_line])
plt.savefig("presentatie/coupled_increasing_3.pdf")
plt.show()

fig, ax = plt.subplots(1,1, figsize=(8,8))
for i in range(0,4):
    ax.plot(axs, delta2[i], '-', color='C' + str(i), label='$\kappa$ = '+str(round(varspace[i],1)))
    ax.hlines(-np.log(0.05)/labda_x2[i], -L,0,ls='--', colors='C'+str(i))
    ax.hlines(-np.log(0.05)/labda_r2[i], a,R,ls='--', colors='C' + str(i))
    # ax.hlines(tx2[i], -L,0,ls='--', colors='C' + str(i))
    # ax.hlines(tr2[i], a,R, ls='--', colors='C' + str(i))
ax.set_xlim(-L, R)
ax.set_ylim(0,t[-1])
ax.set_xlabel(r'$x & r$ (km)')
ax.set_ylabel(r'$t$ (h)')

dt_line = mlines.Line2D([], [], color='k', ls='-', label=r'$t_{ADJ}$')
# dt2_line = mlines.Line2D([], [], color='k', ls='--', label=r'$T_{ADJ}$')
dt3_line = mlines.Line2D([], [], color='k', ls='--', label=r'$1/\lambda_1$')
c_patches = [mpatches.Patch(color='C'+str(j), label=r'$\kappa$ = '+str(round(varspace[j],1))  + ' m$^2$/s' )for j in range(len(varspace))]
ax.legend(handles=c_patches+[dt_line, dt3_line])
plt.savefig("presentatie/coupled_increasing_4.pdf")
plt.show()