#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:09:39 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w

q = pd.read_csv('save_dir/data/20230606_026.csv', delimiter=';', decimal=',', keep_default_na = False)
qnew = q.loc[:,['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'LOCATIE_CODE', 'NUMERIEKEWAARDE']]

MAAS = qnew[qnew['LOCATIE_CODE'] == 'MAASMD'].to_numpy()
BRIEN = qnew[qnew['LOCATIE_CODE'] == 'BRIENOD'].to_numpy()
LOBH = qnew[qnew['LOCATIE_CODE'] == 'LOBH'].to_numpy()



d_old = pd.read_csv('save_dir/data/20230606_020.csv', delimiter=';', decimal=',', keep_default_na = False)
dnew_old = d_old.loc[:,['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'LOCATIE_CODE', 'NUMERIEKEWAARDE', 'BEMONSTERINGSHOOGTE']]

HVH_old = dnew_old[dnew_old['LOCATIE_CODE'] == 'HOEKVHLRTOVR']
LEK_old = dnew_old[dnew_old['LOCATIE_CODE'] == 'LEKHVRTOVR']

HVH900_old = HVH_old[HVH_old['BEMONSTERINGSHOOGTE'] == -900].to_numpy()
HVH450_old = HVH_old[HVH_old['BEMONSTERINGSHOOGTE'] == -450].to_numpy()
HVH250_old = HVH_old[HVH_old['BEMONSTERINGSHOOGTE'] == -250].to_numpy()

LEK700_old = LEK_old[LEK_old['BEMONSTERINGSHOOGTE'] == -700].to_numpy()
LEK500_old = LEK_old[LEK_old['BEMONSTERINGSHOOGTE'] == -500].to_numpy()
LEK250_old = LEK_old[LEK_old['BEMONSTERINGSHOOGTE'] == -250].to_numpy()



d = pd.read_csv('save_dir/data/20230607_014.csv', delimiter=';', decimal=',', keep_default_na = False)
dnew = d.loc[:,['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'LOCATIE_CODE', 'NUMERIEKEWAARDE', 'BEMONSTERINGSHOOGTE']]

HVH = dnew[dnew['LOCATIE_CODE'] == 'HOEKVHLRTOVR']
LEK = dnew[dnew['LOCATIE_CODE'] == 'LEKHVRTOVR']
BRN = dnew[dnew['LOCATIE_CODE'] == 'BRIENOBRTOVR']
KMP = dnew[dnew['LOCATIE_CODE'] == 'KRIMPADIJSLK']
SPK = dnew[dnew['LOCATIE_CODE'] == 'SPIJKNSBWTLK']


HVH900 = HVH[HVH['BEMONSTERINGSHOOGTE'] == -900].to_numpy()
HVH450 = HVH[HVH['BEMONSTERINGSHOOGTE'] == -450].to_numpy()
HVH250 = HVH[HVH['BEMONSTERINGSHOOGTE'] == -250].to_numpy()

LEK700 = LEK[LEK['BEMONSTERINGSHOOGTE'] == -700].to_numpy()
LEK500 = LEK[LEK['BEMONSTERINGSHOOGTE'] == -500].to_numpy()
LEK250 = LEK[LEK['BEMONSTERINGSHOOGTE'] == -250].to_numpy()

BRN650 = BRN[BRN['BEMONSTERINGSHOOGTE'] == -650].to_numpy()
BRN250 = BRN[BRN['BEMONSTERINGSHOOGTE'] == -250].to_numpy()

KMP550 = KMP[KMP['BEMONSTERINGSHOOGTE'] == -550].to_numpy()
KMP400 = KMP[KMP['BEMONSTERINGSHOOGTE'] == -400].to_numpy()

SPK900 = SPK[SPK['BEMONSTERINGSHOOGTE'] == -900].to_numpy()
SPK450 = SPK[SPK['BEMONSTERINGSHOOGTE'] == -450].to_numpy()
SPK250 = SPK[SPK['BEMONSTERINGSHOOGTE'] == -250].to_numpy()


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
Q1_end = 170
Q2_start = Q1_end + 1
Q2_end = 286

Q1 = np.mean(Q[Q1_start:Q1_end+1])
Q2 = np.mean(Q[Q2_start:Q2_end+1])

S1_start = 55
S1_end = Q1_end
S2_start = 190
S2_end = Q2_end
S_LEK_1 = np.mean(S_LEK[S1_start:S1_end+1])
S_LEK_2 = np.mean(S_LEK[S2_start:S2_end+1])
S_BRN_1 = np.mean(S_BRN[S1_start:S1_end+1])
S_BRN_2 = np.mean(S_BRN[S2_start:S2_end+1])

x = np.arange(0,366,1)
fig, axes = plt.subplots(2,1, figsize=(8,8))
ax = axes[0]
ax.plot(x,Q, label='Lobith')
ax.hlines(Q1, Q1_start, Q1_end)
ax.hlines(Q2, Q2_start, Q2_end)
ax.set_xlim(50,350)
ax.set_ylim(500,3000)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('$Q$')
ax.legend()

ax = axes[1]
ax.plot(func(LEK700, per, 0), 'C0-', label='LEK 700')
ax.plot(func(BRN650, per, 0), 'C1-', label='BRN 650')

ax.hlines(S_LEK_1, S1_start, S1_end, color='C0')
ax.hlines(S_LEK_2, S2_start, S2_end, color='C0')
ax.hlines(S_BRN_1, S1_start, S1_end, color='C1')
ax.hlines(S_BRN_2, S2_start, S2_end, color='C1')

ax.set_xlim(50,350)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()
plt.show()


print('LEK e-fold time:', (1-1/np.e)*(S2_start - S1_end))
print('BRN e-fold time:', (1-1/np.e)*(S2_start - S1_end))

fig, axes = plt.subplots(5,1, figsize=(8,8))
ax = axes[0]
ax.plot(func(HVH900, per, 0), '-', label='HVH 900')
ax.plot(func(HVH450, per, 0), '-', label='HVH 450')
ax.plot(func(HVH250, per, 0), '-', label='HVH 250')
ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()

ax = axes[1]
ax.plot(func(LEK700, per, 0), '-', label='LEK 700')
ax.plot(func(LEK500, per, 0), '-', label='LEK 500')
ax.plot(func(LEK250, per, 0), '-', label='LEK 250')
ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()

ax = axes[2]
ax.plot(func(BRN650, per, 0), '-', label='BRN 650')
ax.plot(func(BRN250, per, 0), '-', label='BRN 250')
ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()

ax = axes[3]
ax.plot(func(KMP550, per, 0), '-', label='KMP 550')
ax.plot(func(KMP400, per, 0), '-', label='KMP 400')
ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()

ax = axes[4]
ax.plot(func(SPK900, per, 0), '-', label='SPK 900')
ax.plot(func(SPK450, per, 0), '-', label='SPK 450')
ax.plot(func(SPK250, per, 0), '-', label='SPK 250')
ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()

plt.show()


# Q = 1000
# A = 7500
# L = 40000
# k = 1000

# print(1/ ((Q/A)**2/(4*k) + k*(np.pi/L)**2) /(60*60*24))

def cross(x,y):
    N = len(x)
    res = np.zeros(N)
    for n in range(N):
        res[n] = np.sum(x[:N-n]*y[n:])/(N-n)
    return res

plt.figure()
plt.plot(cross(Q,1/S_LEK))
plt.plot(cross(1/Q,S_LEK))
plt.plot(cross(Q,1/S_BRN))
plt.plot(cross(1/Q,S_BRN))
plt.show()

f = 51

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('day from 01-01-2018')
ax1.set_ylabel('Q', color=color)
ax1.plot(x[f//2:-f//2+1], moving_average(Q,f), color=color, label='Lobith')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:gray'
ax2.set_ylabel(r'$\sigma \sim s$', color=color)  # we already handled the x-label with ax1
ax2.plot(x[f//2:-f//2+1], moving_average(S_LEK,f), color='tab:blue' ,label='LEK')
ax2.plot(x[f//2:-f//2+1], moving_average(S_BRN,f), color='tab:green', label='BRN')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('day from 01-01-2018')
ax1.set_ylabel('Q', color=color)
ax1.plot(x, Q, color=color, label='Lobith')
ax1.plot(x, Q_BRN, color='tab:purple', label='Brien')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:gray'
ax2.set_ylabel(r'$\sigma \sim s$', color=color)  # we already handled the x-label with ax1
ax2.plot(x, S_LEK, color='tab:blue' ,label='LEK')
ax2.plot(x, S_BRN, color='tab:green', label='BRN')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


plt.figure()
plt.plot(np.fft.fft(np.array(func(LEK700, 6, 0), dtype='float')))
plt.plot(np.fft.fft(np.array(func(BRN650, 6, 0), dtype='float')))
plt.show()

plt.figure()
plt.plot(np.fft.fft(np.array(func(LEK700, 24*6, 0), dtype='float')))
plt.plot(np.fft.fft(np.array(func(BRN650, 24*6, 0), dtype='float')))
plt.show()


per = 6
fig, axes = plt.subplots(6,1, figsize=(8,8))
ax = axes[0]
ax.plot(func(HVH900, per, 0), '-', label='HVH 900')
ax.plot(func(HVH450, per, 0), '-', label='HVH 450')
ax.plot(func(HVH250, per, 0), '-', label='HVH 250')
#ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()

ax = axes[1]
ax.plot(func(LEK700, per, 0), '-', label='LEK 700')
ax.plot(func(LEK500, per, 0), '-', label='LEK 500')
ax.plot(func(LEK250, per, 0), '-', label='LEK 250')
#ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()

ax = axes[2]
ax.plot(func(BRN650, per, 0), '-', label='BRN 650')
ax.plot(func(BRN250, per, 0), '-', label='BRN 250')
#ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()

ax = axes[3]
ax.plot(func(KMP550, per, 0), '-', label='KMP 550')
ax.plot(func(KMP400, per, 0), '-', label='KMP 400')
#ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()

ax = axes[4]
ax.plot(func(SPK900, per, 0), '-', label='SPK 900')
ax.plot(func(SPK450, per, 0), '-', label='SPK 450')
ax.plot(func(SPK250, per, 0), '-', label='SPK 250')
#ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()

ax = axes[5]
ax.plot(func(LOBH, per, 0), '-', label='Q')
#ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()

plt.show()
