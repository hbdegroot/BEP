#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:33:43 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data = np.loadtxt('save_dir/data/20230606_026.csv', skiprows=0, usecols=[1])

dataQ = np.genfromtxt('save_dir/data/20230606_026.csv', delimiter=';', skip_header=1)
dataS = np.genfromtxt('save_dir/data/20230606_020.csv', delimiter=';', skip_header=1)


Q = dataQ[:52703,25]

Sb = dataS[:len(Q),25]

for i in range(len(Q)):
    if abs(Q[i]) > 1e10:
        Q[i] = 0
    if abs(Sb[i]) > 1e5:
        Sb[i]=0

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w
        

plt.figure()
plt.plot(moving_average(Q,6))
plt.show()

q = pd.read_csv('save_dir/data/20230606_026.csv', delimiter=';', decimal=',', keep_default_na = False)
qnew = q.loc[:,['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'LOCATIE_CODE', 'NUMERIEKEWAARDE']]

MAAS = qnew[qnew['LOCATIE_CODE'] == 'MAASMD']
BRIEN = qnew[qnew['LOCATIE_CODE'] == 'BRIENOD']
LOBH = qnew[qnew['LOCATIE_CODE'] == 'LOBH']
sLOBH = LOBH[LOBH['NUMERIEKEWAARDE'] <1e10].to_numpy()
LOBH = LOBH[LOBH['NUMERIEKEWAARDE'] <1e10]

MAAS = MAAS.groupby(['WAARNEMINGDATUM']).mean()
BRIEN = BRIEN.groupby(['WAARNEMINGDATUM']).mean()
LOBH = LOBH.groupby(['WAARNEMINGDATUM']).mean()

plt.figure()
plt.plot(MAAS['NUMERIEKEWAARDE'])
plt.plot(BRIEN['NUMERIEKEWAARDE'])
plt.plot(LOBH['NUMERIEKEWAARDE'])
plt.show()



d = pd.read_csv('save_dir/data/20230606_020.csv', delimiter=';', decimal=',', keep_default_na = False)
dnew = d.loc[:,['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'LOCATIE_CODE', 'NUMERIEKEWAARDE', 'BEMONSTERINGSHOOGTE']]

HVH = dnew[dnew['LOCATIE_CODE'] == 'HOEKVHLRTOVR']
LEK = dnew[dnew['LOCATIE_CODE'] == 'LEKHVRTOVR']

HVH900 = HVH[HVH['BEMONSTERINGSHOOGTE'] == -900]
HVH900 = HVH900[HVH900['NUMERIEKEWAARDE'] <1e10].groupby(['WAARNEMINGDATUM']).mean()

HVH450 = HVH[HVH['BEMONSTERINGSHOOGTE'] == -450]
HVH450 = HVH450[HVH450['NUMERIEKEWAARDE'] <1e10].groupby(['WAARNEMINGDATUM']).mean()

HVH250 = HVH[HVH['BEMONSTERINGSHOOGTE'] == -250]
HVH250 = HVH250[HVH250['NUMERIEKEWAARDE'] <1e10].groupby(['WAARNEMINGDATUM']).mean()


LEK700 = LEK[LEK['BEMONSTERINGSHOOGTE'] == -700]
sLEK700 = LEK700[LEK700['NUMERIEKEWAARDE'] <1e10].to_numpy()
LEK700 = LEK700[LEK700['NUMERIEKEWAARDE'] <1e10].groupby(['WAARNEMINGDATUM']).mean()

LEK500 = LEK[LEK['BEMONSTERINGSHOOGTE'] == -500]
sLEK500 = LEK500[LEK500['NUMERIEKEWAARDE'] <1e10].to_numpy()
LEK500 = LEK500[LEK500['NUMERIEKEWAARDE'] <1e10].groupby(['WAARNEMINGDATUM']).mean()

LEK250 = LEK[LEK['BEMONSTERINGSHOOGTE'] == -250]
sLEK250 = LEK250[LEK250['NUMERIEKEWAARDE'] <1e10].to_numpy()
LEK250 = LEK250[LEK250['NUMERIEKEWAARDE'] <1e10].groupby(['WAARNEMINGDATUM']).mean()


plt.figure()
plt.plot(HVH900['NUMERIEKEWAARDE'])
plt.plot(HVH450['NUMERIEKEWAARDE'])
plt.plot(HVH250['NUMERIEKEWAARDE'])
plt.show()

plt.figure()
plt.plot(LEK700['NUMERIEKEWAARDE'])
plt.plot(LEK500['NUMERIEKEWAARDE'])
plt.plot(LEK250['NUMERIEKEWAARDE'])
plt.show()



# plt.figure()
# plt.plot_date(HVH900['WAARNEMINGDATUM'], HVH900['NUMERIEKEWAARDE'])
# plt.plot_date(HVH450['WAARNEMINGDATUM'], HVH450['NUMERIEKEWAARDE'])
# plt.plot_date(HVH250['WAARNEMINGDATUM'], HVH250['NUMERIEKEWAARDE'])
# plt.show()

# plt.figure()
# plt.plot_date(LEK700['WAARNEMINGDATUM'], LEK700['NUMERIEKEWAARDE'])
# plt.plot_date(LEK500['WAARNEMINGDATUM'], LEK500['NUMERIEKEWAARDE'])
# plt.plot_date(LEK250['WAARNEMINGDATUM'], LEK250['NUMERIEKEWAARDE'])
# plt.show()

plt.figure()
plt.plot(moving_average(Q, 24*6)[::24*6])
plt.plot(moving_average(sLOBH[:,3], 24*6)[::24*6])
plt.plot(np.append(np.zeros(31), moving_average(sLEK500[:,3], 6*24)[::24*6]))
plt.plot(np.append(np.zeros(31), moving_average(sLEK700[:,3], 6*24)[::24*6]))
plt.show()

print(LEK['WAARNEMINGDATUM'])
print(q['WAARNEMINGDATUM'])

print(len(LEK250))
