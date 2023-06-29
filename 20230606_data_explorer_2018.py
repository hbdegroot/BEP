#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 19:42:20 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w

q = pd.read_csv('save_dir/data/20230606_026.csv', delimiter=';', decimal=',', keep_default_na = False)

qnew = q.loc[:,['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'LOCATIE_CODE', 'NUMERIEKEWAARDE']]

MAAS_df = qnew[qnew['LOCATIE_CODE'] == 'MAASMD']
BRIEN_df = qnew[qnew['LOCATIE_CODE'] == 'BRIENOD']
LOBH_df = qnew[qnew['LOCATIE_CODE'] == 'LOBH']
MAAS= MAAS_df.to_numpy()
BRIEN = BRIEN_df.to_numpy()
LOBH = LOBH_df.to_numpy()


def replacer(arr):
    for j in range(len(arr)):
        if abs(arr[j]) > 1e5:
            arr[j] = arr[j-1]
    return arr

MAAS = replacer(MAAS[:,3])
BRIEN = replacer(BRIEN[:,3])
LOBH = replacer(LOBH[:,3])

# plt.figure()
# plt.plot(MAAS)
# plt.plot(BRIEN)
# plt.plot(LOBH)
# plt.show()

# plt.figure()
# plt.plot(moving_average(MAAS,24*6)[::24*6])
# plt.plot(moving_average(BRIEN,24*6)[::24*6])
# plt.plot(moving_average(LOBH,24*6)[::24*6])
# plt.show()



d = pd.read_csv('save_dir/data/20230606_020.csv', delimiter=';', decimal=',', keep_default_na = False)
dnew = d.loc[:,['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'LOCATIE_CODE', 'NUMERIEKEWAARDE', 'BEMONSTERINGSHOOGTE']]

HVH = dnew[dnew['LOCATIE_CODE'] == 'HOEKVHLRTOVR']
LEK = dnew[dnew['LOCATIE_CODE'] == 'LEKHVRTOVR']

HVH900 = HVH[HVH['BEMONSTERINGSHOOGTE'] == -900].to_numpy()
HVH450 = HVH[HVH['BEMONSTERINGSHOOGTE'] == -450].to_numpy()
HVH250 = HVH[HVH['BEMONSTERINGSHOOGTE'] == -250].to_numpy()

LEK700 = LEK[LEK['BEMONSTERINGSHOOGTE'] == -700].to_numpy()
LEK500 = LEK[LEK['BEMONSTERINGSHOOGTE'] == -500].to_numpy()
LEK250 = LEK[LEK['BEMONSTERINGSHOOGTE'] == -250].to_numpy()


HVH900 = replacer(HVH900[:,3])
HVH450 = replacer(HVH450[:,3])
HVH250 = replacer(HVH250[:,3])


LEK700 = replacer(LEK700[:,3])
LEK500 = replacer(LEK500[:,3])
LEK250 = replacer(LEK250[:,3])

# plt.figure()
# plt.plot(HVH900)
# plt.plot(HVH250)
# plt.plot(LEK700)
# plt.plot(LEK250)
# plt.show()

# plt.figure()
# plt.plot(moving_average(HVH900,24*6)[::24*6])
# plt.plot(moving_average(HVH250,24*6)[::24*6])
# plt.plot(moving_average(LEK700,24*6)[::24*6])
# plt.plot(moving_average(LEK250,24*6)[::24*6])
# plt.show()



fig, axes = plt.subplots(2,1, figsize=(8,8))

ax = axes[0]
ax.plot(moving_average(MAAS,24*6)[::24*6], label='MAASMOND')
ax.plot(moving_average(BRIEN,24*6)[::24*6], label='BRIENENOORD')
ax.plot(moving_average(LOBH,24*6)[::24*6], label='LOBITH')
ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('$Q$')
ax.legend()

ax = axes[1]
ax.plot(np.append(np.zeros(31),moving_average(HVH900,24*6)[::24*6]), label='HVH -900')
ax.plot(np.append(np.zeros(31),moving_average(HVH250,24*6)[::24*6]), label='HVH -250')
ax.plot(np.append(np.zeros(31),moving_average(LEK700,24*6)[::24*6]), label='LEK -700')
ax.plot(np.append(np.zeros(31),moving_average(LEK250,24*6)[::24*6]), label='LEK -250')
ax.set_xlim(0,365)
ax.set_xlabel('day from 01-01-2018')
ax.set_ylabel('geleidbaarheid')
ax.legend()
plt.show()

def func(arr, period = 24*6, offset=31):
    arr = replacer(arr[:,3])
    return np.append(np.zeros(offset),moving_average(arr,24*6)[::24*6])
    

