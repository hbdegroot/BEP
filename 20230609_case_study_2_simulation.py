#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 09:33:33 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pandas as pd

from scipy.optimize import root
from numpy import pi

from module_nsol_v3 import fd_sol2
from module_nsol_v3 import fd_sol_plus_initial
from module_nsol_v3 import fd_sol2_fw


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w

q = pd.read_csv('save_dir/data/20230606_026.csv', delimiter=';', decimal=',', keep_default_na = False)
qnew = q.loc[:,['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'LOCATIE_CODE', 'NUMERIEKEWAARDE']]

MAAS = qnew[qnew['LOCATIE_CODE'] == 'MAASMD'].to_numpy()
BRIEN = qnew[qnew['LOCATIE_CODE'] == 'BRIENOD'].to_numpy()
LOBH = qnew[qnew['LOCATIE_CODE'] == 'LOBH'].to_numpy()

d = pd.read_csv('save_dir/data/20230607_014.csv', delimiter=';', decimal=',', keep_default_na = False)
dnew = d.loc[:,['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'LOCATIE_CODE', 'NUMERIEKEWAARDE', 'BEMONSTERINGSHOOGTE']]

LEK = dnew[dnew['LOCATIE_CODE'] == 'LEKHVRTOVR']
BRN = dnew[dnew['LOCATIE_CODE'] == 'BRIENOBRTOVR']

LEK700 = LEK[LEK['BEMONSTERINGSHOOGTE'] == -700].to_numpy()
BRN650 = BRN[BRN['BEMONSTERINGSHOOGTE'] == -650].to_numpy()

def replacer(arr):
    for j in range(len(arr)):
        if abs(arr[j]) > 1e5:
            arr[j] = arr[j-1]
    return arr

def func(arr, period = 24*6, offset=0):
    arr = replacer(arr[:,3])
    return np.append(np.zeros(offset),moving_average(arr,period)[::period])
    

per = 24*6    # 1dag, data 10 min
off = 31  # 31dagen

Q = np.array(func(LOBH, per, 0), dtype='float')
Q_BRN = np.array(func(BRIEN, per, 0), dtype='float')
S_LEK = np.array(func(LEK700, per, 0), dtype='float')
S_BRN = np.array(func(BRN650, per, 0), dtype='float')

Q1_start = 55
Q1_end = 173
Q2_start = Q1_end + 1
Q2_end = 286

Q1 = np.mean(Q[Q1_start:Q1_end+1])
Q2 = np.mean(Q[Q2_start:Q2_end+1])
Q1_BRN = np.mean(Q_BRN[Q1_start:Q1_end+1])
Q2_BRN = np.mean(Q_BRN[Q2_start:Q2_end+1])

S1_start = 55
S1_end = Q1_end
S2_start = 190
S2_end = Q2_end
S_LEK_1 = np.mean(S_LEK[S1_start:S1_end+1])
S_LEK_2 = np.mean(S_LEK[S2_start:S2_end+1])
S_BRN_1 = np.mean(S_BRN[S1_start:S1_end+1])
S_BRN_2 = np.mean(S_BRN[S2_start:S2_end+1])

days = np.arange(0,366,1)

fig, axes = plt.subplots(2,1, figsize=(8,8))
ax = axes[0]
ax.plot(days,Q, label='Lobith')
ax.plot(days,Q_BRN, label='BRN')
ax.hlines(Q1, Q1_start, Q1_end, color='C0')
ax.hlines(Q2, Q2_start, Q2_end, color='C0')
ax.hlines(Q1_BRN, Q1_start, Q1_end, color='C1')
ax.hlines(Q2_BRN, Q2_start, Q2_end, color='C1')
ax.set_xlim(50,350)
ax.set_ylim(0,3000)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel(r'$Q$ (m$^3$/s)')
ax.legend()

ax = axes[1]
ax.plot(func(LEK700, per, 0), 'C0-', label='LEK 700')
ax.plot(func(BRN650, per, 0), 'C1-', label='BRN 650')

ax.hlines(S_LEK_1, S1_start, S1_end, color='C0')
ax.hlines(S_LEK_2, S2_start, S2_end, color='C0')
ax.hlines(S_BRN_1, S1_start, S1_end, color='C1')
ax.hlines(S_BRN_2, S2_start, S2_end, color='C1')

ax.set_xlim(50,350)
ax.set_ylim(0,1750)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel(r'$\sigma$ (mS/m)')
ax.legend()
plt.show()

# PARAMETERS

A = 7500
R = 7000
D = 20

a = A/(np.pi*D)

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

SHVH, SLEK, SBRN = 26/30, S_LEK_2/4000, S_BRN_2/4000

k, kappa, L = 3000, 250, 45000
k0 = 850
kappa0 = 250

k_, kappa_, L_ = 700, 250, 100000
k0_ = 800
kappa0_ = 250

x = np.linspace(0,L,100)
x_ = np.linspace(0,L_,100)
r = np.linspace(a,R,100)
axs = np.append(-x[::-1], r)
axs_ = np.append(-x_[::-1], r)

sn = np.append(ssx(x,Q2_BRN,A,k,kappa,R,L,D)[::-1], ssr(r,Q2_BRN,A,k,kappa,R,L,D))
s0 = np.append(ssx(x,Q1_BRN,A,k0,kappa0,R,L,D)[::-1], ssr(r,Q1_BRN,A,k0,kappa0,R,L,D))

sn_ = np.append(ssx(x_,Q2_BRN,A,k_,kappa_,R,L_,D)[::-1], ssr(r,Q2_BRN,A,k_,kappa_,R,L_,D))
s0_ = np.append(ssx(x_,Q1_BRN,A,k0_,kappa0_,R,L_,D)[::-1], ssr(r,Q1_BRN,A,k0_,kappa0_,R,L_,D))

fig, axes = plt.subplots(1,2, figsize=(10,5))
ax = axes[0]
ax.plot(axs, s0, 'C0-', label='best fit')
ax.plot(axs_, s0_, 'C1--', label='long intrusion length')
ax.plot(0,SHVH, 'C2.', markersize=10, label='x=0')
ax.plot(-30000, S_LEK_1/4000, 'C3.', markersize=10, label='LEK')
ax.plot(-40000, S_BRN_1/4000, 'C4.', markersize=10, label='BRN')
ax.set_xlim(-L_,R)
ax.set_ylim(0,1)
ax.set_xlabel('$-x$ & $r$ (m)')
ax.set_ylabel('$s/s_{sea}$')
ax.set_title('initial equillibrium')
ax.legend()

ax = axes[1]
ax.plot(axs, sn, 'C0-', label='best fit')
ax.plot(axs_, sn_, 'C1--', label='long intrusion length')
ax.plot(0,SHVH, 'C2.', markersize=10, label='x=0')
ax.plot(-30000, SLEK, 'C3.', markersize=10, label='LEK')
ax.plot(-40000, SBRN, 'C4.', markersize=10, label='BRN')
ax.set_xlim(-L_,R)
ax.set_ylim(0,1)
ax.set_xlabel('$-x$ & $r$ (m)')
ax.set_ylabel('$s/s_{sea}$')
ax.legend()
ax.set_title('final equillibrium')
plt.show()


temp10 = (Q2_BRN/A)**2/(4*k)
temp20 = k*(pi/L)**2
print(temp10, 1/temp10)
print(temp20, 1/temp20)
print(1/(temp10 + temp20)/(24*3600))

temp10_ = (Q2_BRN/A)**2/(4*k_)
temp20_ = k_*(pi/L_)**2
print(temp10_, 1/temp10_)
print(temp20_, 1/temp20_)
print(1/(temp10_ + temp20_)/(24*3600))


# NUMERIC APPROXIMATION

def Q_t(T):
    if T <= 0: return Q1_BRN
    elif T >= 10*24*3600: return Q2_BRN
    return Q1_BRN + (Q2_BRN - Q1_BRN) *(T/(10*24*3600))

def k_t(T):
    if T <= 0: return k0
    elif T >= 10*24*3600: return k
    return k0 + (k - k0) *(T/(10*24*3600))

def kappa_t(T):
    if T <= 0: return kappa0
    elif T >= 10*24*3600: return kappa
    return kappa0 + (kappa - kappa0) *(T/(10*24*3600))

def k_t_(T):
    if T <= 0: return k0_
    elif T >= 10*24*3600: return k_
    return k0_ + (k_ - k0_) *(T/(10*24*3600))

def kappa_t_(T):
    if T <= 0: return kappa0_
    elif T >= 10*24*3600: return kappa_
    return kappa0_ + (kappa_ - kappa0_) *(T/(10*24*3600))

m = 101
t = np.linspace(0,1.5e6,m)
nx, nr, dt = 1000, 1000, 6000
n_steps = int(t[-1]/dt)


dx = L/(nx+1)
dx_ = L_/(nx+1)
dr = (R-a)/(nr+1)


x = np.linspace(0,L,nx+2)[1:-1]
x_ = np.linspace(0,L_,nx+2)[1:-1]
r = np.linspace(a,R,nr+2)[1:-1]
axs = np.append(-x[::-1], r)
axs_ = np.append(-x_[::-1], r)
sn = np.append(ssx(x,Q2_BRN,A,k,kappa,R,L,D)[::-1], ssr(r,Q2_BRN,A,k,kappa,R,L,D))
s0 = np.append(ssx(x,Q1_BRN,A,k0,kappa0,R,L,D)[::-1], ssr(r,Q1_BRN,A,k0,kappa0,R,L,D))
sn_ = np.append(ssx(x_,Q2_BRN,A,k_,kappa_,R,L_,D)[::-1], ssr(r,Q2_BRN,A,k_,kappa_,R,L_,D))
s0_ = np.append(ssx(x_,Q1_BRN,A,k0_,kappa0_,R,L_,D)[::-1], ssr(r,Q1_BRN,A,k0_,kappa0_,R,L_,D))

# res, sx, sr = fd_sol2(Q_t, A, k, kappa, R, L, D, nx, nr, n_steps, dt, ssx(x,Q1_BRN,A,k0,kappa,R,L,D), ssr(r,Q1_BRN,A,k0,kappa,R,L,D))
res, sx, sr = fd_sol_plus_initial(Q2_BRN, A, k, kappa, R, L, D, nx, nr, n_steps, dt, ssx(x,Q1_BRN,A,k0,kappa0,R,L,D), ssr(r,Q1_BRN,A,k0,kappa0,R,L,D))
res2, sx, sr = fd_sol2(Q_t, A, k_t, kappa_t, R, L, D, nx, nr, n_steps, dt, ssx(x,Q1_BRN,A,k0,kappa0,R,L,D), ssr(r,Q1_BRN,A,k0,kappa0,R,L,D))
res_, sx_, sr_ = fd_sol_plus_initial(Q2_BRN, A, k_, kappa_, R, L_, D, nx, nr, n_steps, dt, ssx(x_,Q1_BRN,A,k0_,kappa0_,R,L_,D), ssr(r,Q1_BRN,A,k0_,kappa0_,R,L_,D))
res2_, sx_, sr_ = fd_sol2(Q_t, A, k_t_, kappa_t_, R, L_, D, nx, nr, n_steps, dt, ssx(x_,Q1_BRN,A,k0_,kappa0_,R,L_,D), ssr(r,Q1_BRN,A,k0_,kappa0_,R,L_,D))

# PLOTTEN

n_skips = int((t[1]-t[0])/dt)
t_num = np.arange(0,(n_steps)*dt, dt)

# fig, axes = plt.subplots(2,1, figsize=(12,12))
# ax = axes[0]
# ax.plot(axs, s0)
# ax.plot(axs, sn)
# ax.set_xlim(-L, R)
# ax.set_ylim(0, 1)
# ax.set_xlabel(r'$-x & r$')
# ax.set_ylabel(r'$s$')

# ax = axes[1]
# ax.set_xlim(0, t[-1])
# # ax.set_ylim(0, ylimit)
# ax.set_xlabel(r'$t$')
# ax.set_ylabel(r'$ds/dx & ds/dr$')

# # Plot the surface.
# def init():
#     ax = axes[0]
#     ax.plot(axs, s0)
#     ax.plot(axs, sn)
#     ax.set_xlim(-L, R)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel(r'$x$')
#     ax.set_ylabel(r'$s$')
    
#     ax = axes[1]
#     ax.plot(axs, s0)
#     ax.plot(axs, sn)
#     ax.set_xlim(-L, R)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel(r'$x$')
#     ax.set_ylabel(r'$s$')
#     return 

# # animation function.  This is called sequentially
# def animate(i):
#     ax = axes[0]
#     ax.clear()
#     ax.plot(axs, s0, ls='--')
#     ax.plot(axs, sn, ls='--')
#     ax.plot(axs, res[i*n_skips], ls='-', label='sim')
#     ax.plot(axs, res2[i*n_skips], ls='-', label='sim2')
#     ax.set_xlim(-L, R)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel(r'$x$')
#     ax.set_ylabel(r'$s$')
#     ax.set_title('t=' + str(t[i]))
#     ax.legend()
    
#     ax = axes[1]
#     ax.clear()
#     ax.plot(axs, s0, ls='--')
#     ax.plot(axs, sn, ls='--')
#     ax.plot(axs, res[i*n_skips], ls='-', label='sim')
#     ax.plot(axs, res2[i*n_skips], ls='-', label='sim2')
#     ax.set_xlim(-L, R)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel(r'$x$')
#     ax.set_ylabel(r'$s$')
#     ax.set_title('t=' + str(t[i]))
#     ax.legend()
#     return 

# anim = FuncAnimation(fig, animate, frames=m-1, interval=10, repeat=False)
# plt.show()

M_days = 20
N_days = 21
t_days = np.arange(-M_days,N_days,1)

s_lek_sim = [res[i][int((1-30000/L)*nx)] for i in range(len(res))]
s_brn_sim = [res[i][int((1-40000/L)*nx)] for i in range(len(res))]
s_lek_sim2 = [res2[i][int((1-30000/L)*nx)] for i in range(len(res))]
s_brn_sim2 = [res2[i][int((1-40000/L)*nx)] for i in range(len(res))]
s_lek_sim_ = [res_[i][int((1-30000/L_)*nx)] for i in range(len(res_))]
s_brn_sim_ = [res_[i][int((1-40000/L_)*nx)] for i in range(len(res_))]
s_lek_sim2_ = [res2_[i][int((1-30000/L_)*nx)] for i in range(len(res_))]
s_brn_sim2_ = [res2_[i][int((1-40000/L_)*nx)] for i in range(len(res_))]

t_hours = np.arange(-M_days*24,N_days*24,1)/24
Q_BRN_hours = np.array(func(BRIEN, 6, 0), dtype='float')
S_LEK_hours = np.array(func(LEK700, 6, 0), dtype='float')
S_BRN_hours = np.array(func(BRN650, 6, 0), dtype='float')


fig, axes = plt.subplots(3,1, figsize=(14,12))
ax = axes[0]
ax.plot(t_days, Q_BRN[Q1_end-M_days:Q1_end+N_days], color='C0', label='daily average')
ax.plot(t_hours, Q_BRN_hours[(Q1_end-M_days)*24:(Q1_end+N_days)*24]/20, color='C0', label='0.05 x hourly average', alpha=0.5)
ax.hlines(Q1_BRN, -M_days, 0, color='C1', label='step simulation\n= long-term average')
ax.vlines(0, Q1_BRN, Q2_BRN, color='C1')
ax.hlines(Q2_BRN, 0, N_days, color='C1')
ax.plot(t_num/86400, [Q_t(T) for T in t_num], color='C2', label='linear simulation')
ax.set_xlim(-M_days, N_days-1)
ax.set_xlabel('day from change')
ax.set_ylabel(r'$Q$ (m$^3$/s) at BRN')
ax.legend()

ax = axes[1]
ax.plot(t_days, S_LEK[Q1_end-M_days:Q1_end+N_days]/4000, color='C0', label='daily average')
ax.plot(t_hours, S_LEK_hours[(Q1_end-M_days)*24:(Q1_end+N_days)*24]/4000, color='C0', label='hourly average', alpha=0.5)
ax.plot(t_num/86400, s_lek_sim, color='C1', ls='-', label='step, best fit')
ax.plot(t_num/86400, s_lek_sim2, color='C2', ls='-', label='linear, best fit')
ax.plot(t_num/86400, s_lek_sim_, color='C3', ls='--', label='step, long intrusion')
ax.plot(t_num/86400, s_lek_sim2_, color='C4', ls='--', label='linear, long intrusion')
ax.hlines(S_LEK_1/4000, -M_days, 0, color='k', label='long-term average')
ax.hlines(S_LEK_2/4000, 0, N_days, color='k')
ax.set_xlim(-M_days, N_days-1)
ax.set_ylim(0, 0.25)
ax.set_xlabel('day from change')
ax.set_ylabel('$s/s_{sea}$ at LEK')
ax.legend()

ax = axes[2]
ax.plot(t_days, S_BRN[Q1_end-M_days:Q1_end+N_days]/4000, color='C0', label='daily average')
ax.plot(t_hours, S_BRN_hours[(Q1_end-M_days)*24:(Q1_end+N_days)*24]/4000, color='C0', label='hourly average', alpha=0.5)
ax.plot(t_num/86400, s_brn_sim, color='C1', ls='-', label='step, best fit')
ax.plot(t_num/86400, s_brn_sim2, color='C2', ls='-', label='linear, best fit')
ax.plot(t_num/86400, s_brn_sim_, color='C3', ls='--', label='step, long intrusion')
ax.plot(t_num/86400, s_brn_sim2_, color='C4', ls='--', label='linear, long intrusion')
ax.hlines(S_BRN_1/4000, -M_days, 0, color='k', label='long-term average')
ax.hlines(S_BRN_2/4000, 0, N_days, color='k')
ax.set_xlim(-M_days, N_days-1)
ax.set_ylim(0, 0.16)
ax.set_xlabel('day from change')
ax.set_ylabel('$s/s_{sea}$ at BRN')
ax.legend()
plt.show()

