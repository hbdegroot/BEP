#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:19:21 2023

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w

q = pd.read_csv('save_dir/data/20230606_056.csv', delimiter=';', decimal=',', keep_default_na = False)
qnew = q.loc[:,['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'LOCATIE_CODE', 'NUMERIEKEWAARDE']]

LOBH_df = qnew[qnew['LOCATIE_CODE'] == 'LOBH']
LOBH = LOBH_df.to_numpy()

d = pd.read_csv('save_dir/data/20230606_057.csv', delimiter=';', decimal=',', keep_default_na = False)
dnew = d.loc[:,['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'LOCATIE_CODE', 'NUMERIEKEWAARDE', 'BEMONSTERINGSHOOGTE']]

HVH_df = dnew[dnew['LOCATIE_CODE'] == 'HOEKVHLRTOVR']
BRN_df = dnew[dnew['LOCATIE_CODE'] == 'BRIENOBRTOVR']

HVH900 = HVH_df[HVH_df['BEMONSTERINGSHOOGTE'] == -900].to_numpy()
HVH250 = HVH_df[HVH_df['BEMONSTERINGSHOOGTE'] == -250].to_numpy()

BRN650 = BRN_df[BRN_df['BEMONSTERINGSHOOGTE'] == -650].to_numpy()
BRN250 = BRN_df[BRN_df['BEMONSTERINGSHOOGTE'] == -250].to_numpy()

def replacer(arr):
    for j in range(len(arr)):
        if abs(arr[j]) > 1e5:
            arr[j] = arr[j-1]
    return arr


def func(arr, period = 24*6, offset=0):
    arr = replacer(arr[:,3])
    return np.append(np.zeros(offset),moving_average(arr,period)[::period])
    

fig, axes = plt.subplots(2,1, figsize=(8,8))

ax = axes[0]
ax.plot(func(LOBH, 24*6, 0), label='Lobith')
ax.set_xlim(0,150)
ax.set_xlabel('day from 01-01-2022')
ax.set_ylabel('$Q$')
ax.legend()

ax = axes[1]
ax.plot(func(HVH900, 24*6, 0), label='HVH 900')
ax.plot(func(HVH250, 24*6, 0), label='HVH 250')
ax.plot(func(BRN650, 24*6, 0), label='BRN 650')
ax.plot(func(BRN250, 24*6, 0), label='BRN 250')
ax.set_xlim(0,150)
ax.set_xlabel('day from 01-01-2022')
ax.set_ylabel('geleidbaarheid')
ax.legend()
plt.show()

    

