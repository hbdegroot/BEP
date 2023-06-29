#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:25:35 2023

@author: hugo

This is a module that takes an array for x, r and t, and the model parameters as input, and returns the result.
"""
import numpy as np
from scipy import integrate
from scipy.special import jv
from scipy.special import yv
from scipy.optimize import root

pi = np.pi

def def_func(default):
    if default == '1':
        Q = 150  # m/s^3
        A = 2000 # m^2
        k = 400
        kappa = 400
        R = 10000
        L = 10000
        D = 1
        good_roots_r = np.array([[3.94030436e-05,7.61848195e-01], #n=0
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
        N_x = 19
        N_r = 19
        return Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r
    
    elif default == '2':
        Q = 150  # m/s^3
        A = 2000 # m^2
        k = 400
        kappa = 400
        R = 10000
        L = 10000
        D = 10
        good_roots_r =np.array([[3.04806792e-05,3.67262193e-01], #n=0
                              [1.41735820e-04,4.52305599e-01], #n=1
                              [3.34234707e-04,5.18957446e-01], #n=2
                              [0.00060759,0.57811936], #n=3
                              [0.00096158,0.63350859], #n=4
                              [0.00139609,0.68693422], #n=5
                              [0.00191101,0.73946396], #n=6
                              [0.00250629,0.79181707], #n=7
                              [0.00318187,0.84453047], #n=8
                              [0.00393772,0.89804038], #n=9
                              [0.00477379,0.95272739], #n=10
                              [0.00569006,1.00894393], #n=11
                              [0.00668651,1.0670328], #n=12
                              [0.00776312,1.12734085], #n=13
                              [0.00891988,1.19023033], #n=14
                              [0.01015676,1.25608918], #n=15
                              [0.01147375,1.32534129], #n=16
                              [0.01287084,1.39845746], #n=17
                              [0.01434803,1.47596781], #n=18
                              [0.01590531,1.55847632], #n=19
                              [0.01754266,1.64667841], #n=20
                              [0.01926008,1.74138267], #n=21
                              [0.02105756,1.84353818], #n=22
                              [0.02293509,1.95426936], #n=23
                              [0.02489268,2.07492117], #n=24
                              [0.02693032,2.20711843]]) #n=25
        N_x = 26
        N_r = 26
        return Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r
    
    elif default == '3':
        Q = 150  # m/s^3
        A = 2000 # m^2
        k = 400
        kappa = 800
        R = 10000
        L = 20000
        D = 5
        good_roots_r =np.array([[6.39175081e-05,4.45405715e-01], #n=0
                              [2.92691289e-04,5.74046877e-01], #n=1
                              [0.00068632,0.68431265], #n=2
                              [0.00124383,0.79027669], #n=3
                              [0.00196472,0.89747847], #n=4
                              [0.00284868,1.00937943], #n=5
                              [0.00389549,1.12887581], #n=6
                              [0.00510502,1.25890398], #n=7
                              [0.00647713,1.40282391], #n=8
                              [0.00801175,1.56478718], #n=9
                              [0.00970881,1.75020302], #n=10
                              [0.01156825,1.96642305], #n=11
                              [0.01359002,2.22383647], #n=12
                              [0.01577409,2.53773672], #n=13
                              [0.01812042,2.9317084 ], #n=14
                              [0.02062898,3.44422888], #n=15
                              [0.02329976,4.14273356], #n=16
                              [0.02613273,5.15729353], #n=17
                              [0.02912788,6.77549078], #n=18
                              [0.03228518,9.78462194], #n=19
                              [0.03560463,17.39142024], #n=20
                              [3.90862088e-02,7.53393908e+01], #n=21
                              [0.04272991,-32.59768504], #n=22
                              [0.04653573,-13.41538096], #n=23
                              [0.05050365,-8.43850453], #n=24
                              [0.05463367,-6.14440994]]) #n=25
        
        N_r = 26
        N_x = 26
        return Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r
    return

def goodrootfinder(N_r, l0, lmax, fine, Q, A, kappa, R, L, D):
    a = A/(pi*D)
    order = Q/(2*kappa*pi*D)
    def phi_r(r,par):
        return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))
    
    phi_bca = lambda rt: phi_r(a,rt)
    phi_bcR = lambda rt: phi_r(R,rt)
    
    def func_r(rt):
        return phi_bca(rt),phi_bcR(rt)
    
    r = np.linspace(a,R,500)
    def n_roots(rt):
        temp = 0
        if abs(phi_r(a,rt))>0.01: return -1
        elif abs(phi_r(R,rt))>0.01: return -1
        for j in range(10,len(r)-20):
            if phi_r(r[j],rt)*phi_r(r[j-1],rt)<0:
                temp+=1
        return temp
    
    roots = [[0,0] for i in range(N_r)]
    
    for i in np.linspace(l0,lmax,fine):
        for j in np.linspace(-10,10,4):
            rt = root(func_r, x0 =(i,j)).x
            N = n_roots(rt)
            #print(rt)
            if N<N_r:
                if N+1>0 and roots[N][0] == 0:
                    roots[N] = rt
        
    mistake = False
    for i in range(len(roots)):
        if roots[i][0] == 0:
            print('Warning: root 0 detected @ N =', i)
            mistake = True
    if mistake:
        for i in range(len(roots)):
            print(i, roots[i])
        raise ValueError('root 0 encountered')
    return roots


def ssx(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, gamma=5.5e-5):
    a = A/(pi*D)
    pwr = Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
    c1 = d1*a**pwr
    c2 = -np.exp(-Q*L/(k*A))*c1
    return c1 * np.exp(-Q*x/(k*A)) + c2
    
def ssr(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, gamma=5.5e-5):
    a = A/(pi*D)
    pwr = Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-Q*L/(k*A)))
    d2 = 1-d1*R**pwr
    return d1*r**pwr + d2
    
def ssx0(x, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, gamma=5.5e-5):
    a = A/(pi*D)
    pwr = 2*Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-2*Q*L/(k*A)))
    c1 = d1*a**pwr
    c2 = -np.exp(-2*Q*L/(k*A))*c1
    return c1 * np.exp(-2*Q*x/(k*A)) + c2

def ssr0(r, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, gamma=5.5e-5):
    a = A/(pi*D)
    pwr = 2*Q/(kappa*pi*D)
    d1 = 1/(R**pwr -a**pwr*np.exp(-2*Q*L/(k*A)))
    d2 = 1-d1*R**pwr
    return d1*r**pwr + d2

def sol(x, r, t, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, gamma=5.5e-5):
    print('Eigen expansion started')
    a = A/(pi*D)

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
    
    def phi_r(r,par):
        return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))

    def phi_r_prime0(r,par):
        c = order
        w = np.sqrt(par[0]/kappa)
        d = par[1]
        
        T1 = d*w*r*yv(c-1, w*r)
        T2 = 2*c*d*yv(c, w*r)
        T3 = -d*w*r*yv(c+1,w*r)
        T4 = w*r*jv(c-1,w*r)
        T5 = 2*c*jv(c,w*r)
        T6 = -w*r*jv(c+1,w*r)
        return 0.5*r**(c-1)*(T1+T2+T3+T4+T5+T6)
    
    def phi_r_dprime0(r,par):
        c = order
        w = np.sqrt(par[0]/kappa)
        d = par[1]
        
        T11 = (c**2+c-w**2*r**2)*yv(c, w*r)/(w**2*r**2)
        T12 = yv(c-1,w*r)/(w*r)
        T1 = d*w**2*r**c*(T11 - T12)
        
        T21 = (c**2+c-w**2*r**2)*jv(c, w*r)/(w**2*r**2)
        T22 = jv(c-1,w*r)/(w*r)
        T2 = w**2*r**c*(T21-T22)
        
        T3 = (c-1)*c*d*r**(c-2)*yv(c,w*r) + c*d*w*r**(c-1)*(yv(c-1,w*r) - yv(c+1,w*r))
        T4 = (c-1)*c*r**(c-2)*jv(c,w*r) + c*w*r**(c-1)*(jv(c-1,w*r) - jv(c+1,w*r))
        return (T1+T2+T3+T4)
        
    def phi_r_n(r,n):
        return phi_r(r, good_roots_r[n])
    
    def phi_r_prime(r,n):
        return phi_r_prime0(r, good_roots_r[n])
    
    def phi_r_dprime(r,n):
        return phi_r_dprime0(r, good_roots_r[n])

    
    def labda_r_n(n):
        return good_roots_r[n][0]

    def phi_x_n(x,n):
        n+=1
        return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)
    
    def phi_x_prime(x,n):
        n+=1
        return n*pi/L
    
    def phi_x_dprime(x,n):
        n+=1
        return -2*n*pi/L*(-Q)/(2*k*A)

    def labda_x_n(n):
        n+=1
        return (Q/A)**2/(4*k) + k*(n*pi/L)**2
    
    def inner_x(n,m):
        return integrate.quad(lambda x: phi_x_n(x,m)*phi_x_n(x,n), 0, L)[0]
    
    def inner_r(n,m):
        return integrate.quad(lambda r: phi_r_n(r,m)*phi_r_n(r,n), a, R)[0]
    
    G_x = np.array([[inner_x(n,m) for n in range(N_x)] for m in range(N_x)])
    G_r = np.array([[inner_r(n,m) for n in range(N_r)] for m in range(N_r)])
    
    labda_x = np.array([labda_x_n(n) for n in range(N_x)])
    labda_r = np.array([labda_r_n(n) for n in range(N_r)])
    
    inv_x = np.linalg.inv(G_x)
    inv_r = np.linalg.inv(G_r)
    
    
    alpha = ssx(0)
    beta = ssx0(0)-ssx(0)
    def f(t):
        return alpha + beta * np.exp(-gamma*t)
    
    f0 = ssx0(0)
    
    
    I1 = [sum([inv_x[n,j]*integrate.quad(lambda x: ssx0(x)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
    I2 = [sum([inv_x[n,j]*integrate.quad(lambda x: (1-x/L)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
    I3 = [sum([inv_x[n,j]*integrate.quad(lambda x: (-Q/(A*L))*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
    
    P0 = [sum([inv_r[n,j]*integrate.quad(lambda r: (r-a)/(R-a)*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
    P1 = [sum([inv_r[n,j]*integrate.quad(lambda r: ssr0(r)*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
    P2 = [sum([inv_r[n,j]*integrate.quad(lambda r: (R-r)/(R-a)*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
    P3 = [sum([inv_r[n,j]*integrate.quad(lambda r: 1/(a-R)*(kappa-Q/(pi*D))/r*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
    
    tempix1 = np.array([I1[n] - f0*I2[n] for n in range(N_x)])
    tempix2 = np.array([-I2[n] for n in range(N_x)])
    tempix3 = np.array([I3[n] for n in range(N_x)])
    
    tempir0 = np.array([-P3[n] for n in range(N_r)])
    tempir1 = np.array([-P0[n] + P1[n] - f0*P2[n] + P3[n] for n in range(N_r)])
    tempir2 = np.array([-P2[n] for n in range(N_r)])
    tempir3 = np.array([P3[n] for n in range(N_r)])
    
    
    tempitx1 = np.array([np.exp(-labda_x[n]*t) for n in range(N_x)])
    tempitx2 = np.array([-beta*gamma/(labda_x[n]-gamma)*(np.exp(-gamma*t) - np.exp(-labda_x[n]*t)) for n in range(N_x)])
    tempitx3 = np.array([alpha/labda_x[n]*(1-np.exp(-labda_x[n]*t)) + beta/(labda_x[n]-gamma)*(np.exp(-gamma*t) - np.exp(-labda_x[n]*t)) for n in range(N_x)])
    
    tempitr0 = np.array([1/labda_r[n]*(1-np.exp(-labda_r[n]*t)) for n in range(N_r)])
    tempitr1 = np.array([np.exp(-labda_r[n]*t) for n in range(N_r)])
    tempitr2 = np.array([-beta*gamma/(labda_r[n]-gamma)*(np.exp(-gamma*t) - np.exp(-labda_r[n]*t)) for n in range(N_r)])
    tempitr3 = np.array([alpha/labda_r[n]*(1-np.exp(-labda_r[n]*t)) + beta/(labda_r[n]-gamma)*(np.exp(-gamma*t) - np.exp(-labda_r[n]*t)) for n in range(N_r)])
    
    Tx = np.zeros((len(t), N_x))
    Tr = np.zeros((len(t), N_r))
    
    for i in range(len(t)):
        T1 = tempitx1[:,i] * tempix1
        T2 = tempitx2[:,i] * tempix2
        T3 = tempitx3[:,i] * tempix3
        Tx[i,:] = T1 + T2 + T3
        
    for i in range(len(t)):
        T0 = tempitr0[:,i] * tempir0
        T1 = tempitr1[:,i] * tempir1
        T2 = tempitr2[:,i] * tempir2
        T3 = tempitr3[:,i] * tempir3
        Tr[i,:] = T0 + T1 + T2 + T3
        
        
    def ksi(x,t):
        return f(t)*(1-x/L)
    
    def psi(r,t):
        return (f(t)-1)*r/(a-R)+(R*f(t)-a)/(R-a)
    
    KSI = np.zeros((len(t), len(x)))
    PSI = np.zeros((len(t), len(r)))
    for i in range(len(t)):
        for j in range(len(x)):
            KSI[i,j] = ksi(x[j], t[i])
    
    for i in range(len(t)):
        for j in range(len(r)):
            PSI[i,j] = psi(r[j], t[i])
            
    solx = Tx @ np.array([phi_x_n(x,n) for n in range(N_x)]) + KSI
    solr = Tr @ np.array([phi_r_n(r,n) for n in range(N_r)]) + PSI

            
    result = [np.zeros(10) for i in range(len(t))]
    for i in range(len(t)):
        result[i] = np.append(solx[i][::-1], solr[i])
        
    def ksi_dx(x,t):
        return -f(t)/L
    def psi_dr(r,t):
        return (f(t)-1)/(a-R)
    
    KSI_dx = np.zeros(len(t))
    PSI_dr = np.zeros(len(t))
    for i in range(len(t)):
        KSI_dx[i] = ksi_dx(0, t[i])
    
    for i in range(len(t)):
        PSI_dr[i] = psi_dr(a, t[i])
        
    dsdx = Tx @ np.array([phi_x_prime(0,n) for n in range(N_x)]) + KSI_dx
    dsdr = Tr @ np.array([phi_r_prime(a,n) for n in range(N_r)]) + PSI_dr
    
    ds2dx2 = Tx @ np.array([phi_x_dprime(0,n) for n in range(N_x)])
    ds2dr2 = Tr @ np.array([phi_r_dprime(a,n) for n in range(N_r)])
    
    transx = k*ds2dx2 + Q/A*dsdx
    transr = kappa*ds2dr2 + 1/a*(kappa-Q/(pi*D))*dsdr
    
    def sx(arr):
        n = len(x)
        dx = x[n-2] - x[n-1]
        dsdx = (arr[n-3] - arr[n-1])/(2*dx)
        ds2dx2 = (arr[n-3] -2* arr[n-2] + arr[n-1])/dx**2
        return dsdx
        return k*ds2dx2 + Q/A*dsdx
    
    def sr(arr):
        n = len(x)
        dr = r[1] - r[0]
        dsdr = (arr[n+2] - arr[n])/(2*dr)
        ds2dr2 = (arr[n+2] - 2*arr[n+1] + arr[n])/dr**2
        return dsdr
        return kappa*ds2dr2 + 1/a*(kappa-Q/(pi*D))*dsdr
    
    sx_arr = np.array([sx(result[i]) for i in range(len(result))])
    sr_arr = np.array([sr(result[i]) for i in range(len(result))])
    
    dt = t[1] - t[0]
    sxt_arr = np.array([0] + [(result[i][len(x)-1]-result[i-1][len(x)-1])/dt for i in range(1,len(result))])
    srt_arr = np.array([0] + [(result[i][len(x)]-result[i-1][len(x)])/dt for i in range(1,len(result))])
    
    
    print('Eigen expansion finished \n')
    
    #return result, sxt_arr, srt_arr, transx, transr
    return result, dsdx, dsdr, sx_arr, sr_arr
    return result, dsdx, dsdr, transx, transr


def sol_numf(x, r, t, Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r, t_num, f_num, dt):
    print('Eigen expansion started')
    a = A/(pi*D)

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
    
    def phi_r(r,par):
        return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))

    def phi_r_prime0(r,par):
        c = order
        w = np.sqrt(par[0]/kappa)
        d = par[1]
        
        T1 = d*w*r*yv(c-1, w*r)
        T2 = 2*c*d*yv(c, w*r)
        T3 = -d*w*r*yv(c+1,w*r)
        T4 = w*r*jv(c-1,w*r)
        T5 = 2*c*jv(c,w*r)
        T6 = -w*r*jv(c+1,w*r)
        return 0.5*r**(c-1)*(T1+T2+T3+T4+T5+T6)
    
    def phi_r_dprime0(r,par):
        c = order
        w = np.sqrt(par[0]/kappa)
        d = par[1]
        
        T11 = (c**2+c-w**2*r**2)*yv(c, w*r)/(w**2*r**2)
        T12 = yv(c-1,w*r)/(w*r)
        T1 = d*w**2*r**c*(T11 - T12)
        
        T21 = (c**2+c-w**2*r**2)*jv(c, w*r)/(w**2*r**2)
        T22 = jv(c-1,w*r)/(w*r)
        T2 = w**2*r**c*(T21-T22)
        
        T3 = (c-1)*c*d*r**(c-2)*yv(c,w*r) + c*d*w*r**(c-1)*(yv(c-1,w*r) - yv(c+1,w*r))
        T4 = (c-1)*c*r**(c-2)*jv(c,w*r) + c*w*r**(c-1)*(jv(c-1,w*r) - jv(c+1,w*r))
        return (T1+T2+T3+T4)
        
    def phi_r_n(r,n):
        return phi_r(r, good_roots_r[n])
    
    def phi_r_prime(r,n):
        return phi_r_prime0(r, good_roots_r[n])
    
    def phi_r_dprime(r,n):
        return phi_r_dprime0(r, good_roots_r[n])

    
    def labda_r_n(n):
        return good_roots_r[n][0]

    def phi_x_n(x,n):
        n+=1
        return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)
    
    def phi_x_prime(x,n):
        n+=1
        return n*pi/L
    
    def phi_x_dprime(x,n):
        n+=1
        return -2*n*pi/L*(-Q)/(2*k*A)

    def labda_x_n(n):
        n+=1
        return (Q/A)**2/(4*k) + k*(n*pi/L)**2
    
    def inner_x(n,m):
        return integrate.quad(lambda x: phi_x_n(x,m)*phi_x_n(x,n), 0, L)[0]
    
    def inner_r(n,m):
        return integrate.quad(lambda r: phi_r_n(r,m)*phi_r_n(r,n), a, R)[0]
    
    G_x = np.array([[inner_x(n,m) for n in range(N_x)] for m in range(N_x)])
    G_r = np.array([[inner_r(n,m) for n in range(N_r)] for m in range(N_r)])
    
    labda_x = np.array([labda_x_n(n) for n in range(N_x)])
    labda_r = np.array([labda_r_n(n) for n in range(N_r)])
    
    inv_x = np.linalg.inv(G_x)
    inv_r = np.linalg.inv(G_r)
    
    f0 = ssx0(f_num[0])
    
    
    I1 = [sum([inv_x[n,j]*integrate.quad(lambda x: ssx0(x)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
    I2 = [sum([inv_x[n,j]*integrate.quad(lambda x: (1-x/L)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
    I3 = [sum([inv_x[n,j]*integrate.quad(lambda x: (-Q/(A*L))*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
    
    P0 = [sum([inv_r[n,j]*integrate.quad(lambda r: (r-a)/(R-a)*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
    P1 = [sum([inv_r[n,j]*integrate.quad(lambda r: ssr0(r)*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
    P2 = [sum([inv_r[n,j]*integrate.quad(lambda r: (R-r)/(R-a)*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
    P3 = [sum([inv_r[n,j]*integrate.quad(lambda r: 1/(a-R)*(kappa-Q/(pi*D))/r*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
    
    tempix1 = np.array([I1[n] for n in range(N_x)])
    tempix2 = np.array([-I2[n] for n in range(N_x)])
    tempix3 = np.array([I3[n] + labda_x[n]*I2[n] for n in range(N_x)])
    
    tempir0 = np.array([-P3[n] for n in range(N_r)])
    tempir1 = np.array([-P0[n] + P1[n] for n in range(N_r)])
    tempir2 = np.array([-P2[n] for n in range(N_r)])
    tempir3 = np.array([P3[n] + labda_r[n]*P2[n] for n in range(N_r)])
    
    #trapz = integrate.cumulative_trapezoidal(f_num, t_num)
    def trapsx(t,n):
        res = []
        for p in t:
            index = int(p//dt)
            t_new = t_num[index] + p%dt
            if index < len(t_num)-2:
                f_new = f_num[index] + (p%dt)/dt*(f_num[index+1]-f_num[index])
            else: f_new = f_num[index-1]
            tarr = np.append(t_num[:index], np.array([t_new]))
            farr = np.append(f_num[:index], np.array([f_new])) * np.exp(labda_x[n]*(tarr-p))
            res.append(integrate.trapz(farr, tarr))
        return np.array(res)

    def trapsr(t,n):
        res = []
        for p in t:
            index = int(p//dt)
            t_new = t_num[index] + p%dt
            if index < len(t_num)-2:
                f_new = f_num[index] + (p%dt)/dt*(f_num[index+1]-f_num[index])
            else: f_new = f_num[index-1]
            tarr = np.append(t_num[:index], np.array([t_new]))
            farr = np.append(f_num[:index], np.array([f_new])) * np.exp(labda_r[n]*(tarr-p))
            res.append(integrate.trapz(farr, tarr))
        return np.array(res)
    
    def f(t):
        res = []
        for p in t:
            index = int(p//dt)
            if index < len(t_num)-2:
                res.append(f_num[index] + (p%dt)/dt*(f_num[index+1]-f_num[index]))
            else: res.append(f_num[index-1])
        return np.array(res)
        
    tempitx1 = np.array([np.exp(-labda_x[n]*t) for n in range(N_x)])
    tempitx2 = np.array([f(t) for n in range(N_x)])
    tempitx3 = np.array([trapsx(t,n) for n in range(N_x)])
    
    tempitr0 = np.array([1/labda_r[n]*(1-np.exp(-labda_r[n]*t)) for n in range(N_r)])
    tempitr1 = np.array([np.exp(-labda_r[n]*t) for n in range(N_r)])
    tempitr2 = np.array([f(t) for n in range(N_r)])
    tempitr3 = np.array([trapsr(t,n) for n in range(N_r)])
    
    Tx = np.zeros((len(t), N_x))
    Tr = np.zeros((len(t), N_r))
    
    for i in range(len(t)):
        T1 = tempitx1[:,i] * tempix1
        T2 = tempitx2[:,i] * tempix2
        T3 = tempitx3[:,i] * tempix3
        Tx[i,:] = T1 + T2 + T3
        
    for i in range(len(t)):
        T0 = tempitr0[:,i] * tempir0
        T1 = tempitr1[:,i] * tempir1
        T2 = tempitr2[:,i] * tempir2
        T3 = tempitr3[:,i] * tempir3
        Tr[i,:] = T0 + T1 + T2 + T3
        
        
    def ksi(x,t):
        return f(t)*(1-x/L)
    
    def psi(r,t):
        return (f(t)-1)*r/(a-R)+(R*f(t)-a)/(R-a)
    
    KSI = np.zeros((len(t), len(x)))
    PSI = np.zeros((len(t), len(r)))
    
    for j in range(len(x)):
        KSI[:,j] = ksi(x[j], t)
    
    
    for j in range(len(r)):
        PSI[:,j] = psi(r[j], t)
            
    solx = Tx @ np.array([phi_x_n(x,n) for n in range(N_x)]) + KSI
    solr = Tr @ np.array([phi_r_n(r,n) for n in range(N_r)]) + PSI

            
    result = [np.zeros(10) for i in range(len(t))]
    for i in range(len(t)):
        result[i] = np.append(solx[i][::-1], solr[i])
        
    def ksi_dx(x,t):
        return -f(t)/L
    def psi_dr(r,t):
        return (f(t)-1)/(a-R)
    
    KSI_dx = np.zeros(len(t))
    PSI_dr = np.zeros(len(t))

    KSI_dx = ksi_dx(0, t)
    PSI_dr = psi_dr(a, t)
        
    dsdx = Tx @ np.array([phi_x_prime(0,n) for n in range(N_x)]) + KSI_dx
    dsdr = Tr @ np.array([phi_r_prime(a,n) for n in range(N_r)]) + PSI_dr
    
    ds2dx2 = Tx @ np.array([phi_x_dprime(0,n) for n in range(N_x)])
    ds2dr2 = Tr @ np.array([phi_r_dprime(a,n) for n in range(N_r)])
    
    transx = k*ds2dx2 + Q/A*dsdx
    transr = kappa*ds2dr2 + 1/a*(kappa-Q/(pi*D))*dsdr
    
    def sx(arr):
        n = len(x)
        dx = x[n-2] - x[n-1]
        dsdx = (arr[n-3] - arr[n-1])/(2*dx)
        ds2dx2 = (arr[n-3] -2* arr[n-2] + arr[n-1])/dx**2
        return dsdx
        return k*ds2dx2 + Q/A*dsdx
    
    def sr(arr):
        n = len(x)
        dr = r[1] - r[0]
        dsdr = (arr[n+2] - arr[n])/(2*dr)
        ds2dr2 = (arr[n+2] - 2*arr[n+1] + arr[n])/dr**2
        return dsdr
        return kappa*ds2dr2 + 1/a*(kappa-Q/(pi*D))*dsdr
    
    sx_arr = np.array([sx(result[i]) for i in range(len(result))])
    sr_arr = np.array([sr(result[i]) for i in range(len(result))])
    
    dt = t[1] - t[0]
    sxt_arr = np.array([0] + [(result[i][len(x)-1]-result[i-1][len(x)-1])/dt for i in range(1,len(result))])
    srt_arr = np.array([0] + [(result[i][len(x)]-result[i-1][len(x)])/dt for i in range(1,len(result))])
    
    
    print('Eigen expansion finished \n')
    
    #return result, sxt_arr, srt_arr, transx, transr
    return result, dsdx, dsdr, sx_arr, sr_arr

def tc(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r):
    N_r += 1
    N_x += 1
    pi = np.pi
    
    a = A/(pi*D)
    
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
    
    def phi_r(r,par):
        return (r)**(order)*jv(order, np.sqrt(par[0]/kappa)*(r)) + par[1]*(r)**(order)*yv(order, np.sqrt(par[0]/kappa)*(r))
    
    def phi_r_prime0(r,par):
        c = order
        w = np.sqrt(par[0]/kappa)
        d = par[1]
        
        T1 = d*w*r*yv(c-1, w*r)
        T2 = 2*c*d*yv(c, w*r)
        T3 = -d*w*r*yv(c+1,w*r)
        T4 = w*r*jv(c-1,w*r)
        T5 = 2*c*jv(c,w*r)
        T6 = -w*r*jv(c+1,w*r)
        return 0.5*r**(c-1)*(T1+T2+T3+T4+T5+T6)
    
    def phi_r_dprime0(r,par):
        c = order
        w = np.sqrt(par[0]/kappa)
        d = par[1]
        
        T11 = (c**2+c-w**2*r**2)*yv(c, w*r)/(w**2*r**2)
        T12 = yv(c-1,w*r)/(w*r)
        T1 = d*w**2*r**c*(T11 - T12)
        
        T21 = (c**2+c-w**2*r**2)*jv(c, w*r)/(w**2*r**2)
        T22 = jv(c-1,w*r)/(w*r)
        T2 = w**2*r**c*(T21-T22)
        
        T3 = (c-1)*c*d*r**(c-2)*yv(c,w*r) + c*d*w*r**(c-1)*(yv(c-1,w*r) - yv(c+1,w*r))
        T4 = (c-1)*c*r**(c-2)*jv(c,w*r) + c*w*r**(c-1)*(jv(c-1,w*r) - jv(c+1,w*r))
        return (T1+T2+T3+T4)
    
    
    #We now count from n=0 for simplicity and to add confusion :)
    
    def phi_r_n(r,n):
        if n==0: return ssr(r)
        return phi_r(r, good_roots_r[n-1])
    
    def labda_r_n(n):
        if n==0: return 0
        return good_roots_r[n-1][0]
    
    # def phi_r_prime(r,n):
    #     return phi_r_prime0(r, good_roots_r[n])
    
    # def phi_r_dprime(r,n):
    #     return phi_r_dprime0(r, good_roots_r[n])
    
    
    def phi_x_n(x,n):
        if n==0: return ssx(x)
        return np.exp(-Q/(2*k*A)*x)*np.sin(n*pi*x/L)
    
    def labda_x_n(n):
        if n==0: return 0
        return (Q/A)**2/(4*k) + k*(n*pi/L)**2
    
    # def phi_x_prime(x,n):
    #     n+=1
    #     return n*pi/L
    
    # def phi_x_dprime(x,n):
    #     n+=1
    #     return -2*n*pi/L*(-Q)/(2*k*A)
    
    def inner_x(n,m):
        return integrate.quad(lambda x: phi_x_n(x,m)*phi_x_n(x,n), 0, L)[0]
    
    def inner_r(n,m):
        return integrate.quad(lambda r: phi_r_n(r,m)*phi_r_n(r,n), a, R)[0]
    
    G_x = np.array([[inner_x(n,m) for n in range(N_x)] for m in range(N_x)])
    G_r = np.array([[inner_r(n,m) for n in range(N_r)] for m in range(N_r)])
    
    labda_x = np.array([labda_x_n(n) for n in range(N_x)])
    labda_r = np.array([labda_r_n(n) for n in range(N_r)])
    
    inv_x = np.linalg.inv(G_x)
    inv_r = np.linalg.inv(G_r)
    
    
    I1 = [sum([inv_x[n,j]*integrate.quad(lambda x: ssx0(x)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
    I2 = [sum([inv_x[n,j]*integrate.quad(lambda x: A/Q*(x-L)*phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
    I3 = [sum([inv_x[n,j]*integrate.quad(lambda x: phi_x_n(x,j), 0, L)[0] for j in range(N_x)]) for n in range(N_x)]
    
    P1 = [sum([inv_r[n,j]*integrate.quad(lambda r: (ssr0(r)-1)*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
    P2 = [sum([inv_r[n,j]*integrate.quad(lambda r: -a*(r-R)/(kappa-Q/(pi*D))*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
    P3 = [sum([inv_r[n,j]*integrate.quad(lambda r: a/r*phi_r_n(r,j), a, R)[0] for j in range(N_r)]) for n in range(N_r)]
    
    alpha = P3[0]*phi_r_n(a,0) - I3[0]*phi_x_n(0,0)
    beta = P2[0]*phi_r_n(a,0) + I2[0]*phi_x_n(0,0) + a*(a-R)/(kappa-Q/(pi*D)) + A*L/Q
    
    return alpha/beta


def eigenvalues_x(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r):
    def labda_x_n(n):
        n+=1
        return (Q/A)**2/(4*k) + k*(n*pi/L)**2
    return np.array([labda_x_n(n) for n in range(N_x)])

def eigenvalues_r(Q, A, k, kappa, R, L, D, good_roots_r, N_x, N_r):
    def labda_r_n(n):
        return good_roots_r[n][0]
    return np.array([labda_r_n(n) for n in range(N_r)])
    