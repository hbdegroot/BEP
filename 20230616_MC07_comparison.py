#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:57:05 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import integrate

pi = np.pi
f = 1
perc = 1/np.e

xi = np.linspace(0,1,50)
tau = np.linspace(0,0.5,200)

N = 30

def dmlts(Pe, multiplier=2):
    def ss(xi):
        temp = np.exp(-Pe)
        return f/(1-temp)*(np.exp(-Pe*xi) - temp)
    
    def h1(xi):
        P = Pe/multiplier
        temp = np.exp(-P)
        return f/(1-temp)*(np.exp(-P*xi) - temp)
    
    def h2(xi):
        P = Pe*multiplier
        temp = np.exp(-P)
        return f/(1-temp)*(np.exp(-P*xi) - temp)
    
    def phi_n(xi,n):
        n+=1
        return np.exp(-Pe*xi/2)*np.sin(n*pi*xi)
    
    def labda_n(n):
        n+=1
        return Pe**2/4 + n**2 * pi**2
    
    def inner(n,m):
        return integrate.quad(lambda xi: phi_n(xi,m)*phi_n(xi,n), 0, 1)[0]
    
    G = np.array([[inner(n,m) for n in range(N)] for m in range(N)])
    
    labda = np.array([labda_n(n) for n in range(N)])
    
    inv = np.linalg.inv(G)
    
    def solf(h):
        I1 = [sum([inv[n,j]*integrate.quad(lambda x: (h(x)-f*(1-x))*phi_n(x,j), 0, 1)[0] for j in range(N)]) for n in range(N)]
        I2 = [sum([inv[n,j]*integrate.quad(lambda x: (-Pe*f)*phi_n(x,j), 0, 1)[0] for j in range(N)]) for n in range(N)]
        
        tempi1 = np.array([np.exp(-labda[n]*tau) for n in range(N)])
        tempi2 = np.array([(1-np.exp(-labda[n]*tau))/labda[n] for n in range(N)])
        
        T = np.zeros((len(tau), N))
        
        for i in range(len(tau)):
            T1 = tempi1[:,i] * I1
            T2 = tempi2[:,i] * I2
            T[i,:] = T1 + T2
          
        def ksi(xi):
            return f*(1-xi)
        
        KSI = np.zeros((len(tau), len(xi)))
        for i in range(len(tau)):
            for j in range(len(xi)):
                KSI[i,j] = ksi(xi[j])
        
        sol = T @ np.array([phi_n(xi,n) for n in range(N)]) + KSI
    
        s0 = h(xi)
        s9 = ss(xi)
    
        deltas = s9 - s0
        deltat = np.zeros(len(xi))
        if np.sum(deltas)  < 0:
            for i in range(len(xi)):
                for j in range(len(tau)):
                    if s9[i] - sol[j][i] >= deltas[i]*perc and deltat[i] == 0:
                        deltat[i] = tau[j]
        else:
            for i in range(len(xi)):
                for j in range(len(tau)):
                    if s9[i] - sol[j][i] <= deltas[i]*perc and deltat[i] == 0:
                        deltat[i] = tau[j]
        dt = np.max(deltat[1:-1])       
        return sol, dt, deltat

    sol1, dt1, deltat1 = solf(h1)
    sol2, dt2, deltat2 = solf(h2)
    T_labda = -np.log(perc)/labda

    return dt1, dt2, deltat1, deltat2


Pe = np.logspace(-1,1,11)
dt1, dt2 = np.zeros(len(Pe)), np.zeros(len(Pe))
deltat1, deltat2 = [np.zeros(len(xi)) for P in Pe], [np.zeros(len(xi)) for P in Pe]


for i in range(len(Pe)):
    print('Calculation ' + str(i) + '/' + str(len(Pe)) + ' strated...')
    dt1[i], dt2[i], deltat1[i], deltat2[i] = dmlts(Pe[i],4)
print('Finished')
    
Pplot = np.linspace(0.001, 100, 1000)
    
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.scatter(Pe, dt2, marker='^')
ax.scatter(Pe, dt1, marker='v')
ax.plot(np.linspace(0,2*Pe[-1],100), 1/(np.linspace(0,2*Pe[-1],100)**2/4 + pi**2), 'k-', label=r'$1/\Lambda_1$')
ax.plot(Pplot, 1/(2*Pplot), label='MC07')
ax.set_xlim(0,Pe[-1]+1)
ax.set_ylim(0,0.5)
ax.set_xlabel(r'Pe')
ax.set_ylabel(r'$\mathcal{T}_{ADJ}$')
ax.legend()
plt.show()


c = []
for j in range(len(Pe)):
    c.append([np.log10(Pe[j])]*len(xi))
    
fig, axes = plt.subplots(1,2, figsize=(15,6))
ax = axes[0]
im = ax.scatter([xi for j in range(len(Pe))], [deltat2[j] for j in range(len(Pe))],c=c, cmap='viridis', marker='^', label='increasing $s$')
ax.scatter([np.linspace(0,1,len(Pplot)) for P in Pe], [[1/(2*Pe[j]) for P in Pplot] for j in range(len(Pe))], c=[[P]*len(Pplot) for P in Pe], cmap='viridis', marker = '.', alpha = 1)
ax.set_xlim(0,1)
ax.set_ylim(0,0.2)
ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'$\tau_{ADJ}$')
plt.colorbar(im, ax=ax, label=r'$^{10}\log(\mathrm{Pe})$')
ax.legend()

ax = axes[1]
im = ax.scatter([xi for j in range(len(Pe))], [deltat1[j] for j in range(len(Pe))],c=c, cmap='viridis', marker='v', label='decreasing $s$')
ax.scatter([np.linspace(0,1,len(Pplot)) for P in Pe], [[1/(2*Pe[j]) for P in Pplot] for j in range(len(Pe))], c=[[P]*len(Pplot) for P in Pe], cmap='viridis', marker = '.', alpha = 1)
ax.set_xlim(0,1)
ax.set_ylim(0,0.2)
ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'$\tau_{ADJ}$')
plt.colorbar(im, ax=ax, label=r'$^{10}\log(\mathrm{Pe})$')
ax.legend()
plt.show()


# # PARAMETERISATION FIT COMMENTED
# ## parametrisation
# from scipy.optimize import curve_fit

# def l(M,a,b,c):
#     x,y = M
#     return 1/(a*x**b*y**c + pi**2)

# Np = len(Pe)
# Nm = len(mspace)
# xdata = np.zeros((2,Np*Nm))
# temp=0
# for i in range(Np):
#     for j in range(Nm):
#         xdata[0,temp] = Pe[i]
#         xdata[1,temp] = mspace[j]
#         temp+=1

# ydata = []
# for i in range(Np):
#     for j in range(Nm):
#         ydata.append(ds[j][i])

# popt, pcov = curve_fit(l, xdata, np.array(ydata), p0=[0,0,0])


# print('a =', popt[0], '+/-', np.sqrt(pcov[0,0]))
# print('b =', popt[1], '+/-', np.sqrt(pcov[1,1]))
# print('c =', popt[2], '+/-', np.sqrt(pcov[2,2]))
# # print('d =', popt[3], '+/-', np.sqrt(pcov[3,3]))
# # print('e =', popt[4], '+/-', np.sqrt(pcov[4,4]))
# # print('f =', popt[5], '+/-', np.sqrt(pcov[5,5]))

# def _l(P,q,*args):
#     return l((P,q), *args)

# plt.figure()
# c=0
# for i in range(0,Np,Np//Np):
#     plt.plot(mspace, [ds[j][i] for j in range(Nm)], '.', color='C'+str(c), label='Pe = ' + str(round(Pe[i],3)))
#     plt.plot(mspace, _l(Pe[i],mspace, *popt), color='C'+str(c))
#     c+=1
# plt.xlabel('q')
# plt.ylabel('$\Lambda$')
# plt.legend()
# plt.show()

# plt.figure()
# c=0
# for j in range(0,Nm,Nm//Nm):
#     plt.plot(Pe,ds[:][j], '.', color = 'C' +str(c), label='m = ' +str(round(mspace[j],3)))
#     plt.plot(Pe, _l(Pe,mspace[j], *popt), color = 'C' +str(c))
#     c+=1
# plt.xlabel('P')
# plt.ylabel('$\Lambda$')
# plt.legend()
# plt.show()



# # labda = np.transpose(np.array(ds1[:Nm//2]))
# # q = mspace[:Nm//2]
# # P = Pe
# # Nq = Nm//2

# # fig, axes = plt.subplots(1,3, figsize=(16,4))
# # fig.suptitle(str(mspace[0]) + ' < m < ' + str(mspace[-1]))

# # ax = axes[0]
# # im = ax.imshow(labda[::-1,:], extent=[q[0],q[-1], P[0],P[-1]], aspect='auto')
# # ax.contour(q,P,labda, 20, colors='k', inline=True)
# # ax.set_xlabel('q')
# # ax.set_ylabel('P')
# # plt.colorbar(im, ax=ax)

# # ax = axes[1]
# # c=0
# # for i in range(0,Np,Np//5):
# #     ax.plot(q, labda[i,:], '.', color='C'+str(c), label='P = ' + str(round(P[i],3)))
# #     ax.plot(q, _l(P[i],q, *popt), color='C'+str(c))
# #     c+=1
# # ax.set_xlabel('q')
# # ax.set_ylabel('$\Lambda$')
# # ax.legend()

# # ax = axes[2]
# # c=0
# # for j in range(0,Nq, Nq//5):
# #     ax.plot(P,labda[:,j], '.', color = 'C' +str(c), label='q = ' +str(round(q[j],3)))
# #     ax.plot(P, _l(P,q[j], *popt), color = 'C' +str(c))
# #     c+=1
# # ax.set_xlabel('P')
# # ax.set_ylabel('$\Lambda$')
# # ax.legend()

# # #plt.savefig('save_dir/figures/qlarge')

# # plt.show()