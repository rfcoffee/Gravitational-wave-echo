#!/usr/bin/env python
# coding: utf-8

# # Strain-time and potential from potchl-teller potential


# Import libraries

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

from scipy.special import gamma
from numpy import exp,conjugate,real,arcsinh,cosh

# lambda satisfies equation: lambda*(1+lambda)+l(1+l)=0, where l is angular momentum quantum number, we take l=2
lambdar=-1/2;
lambdai=math.sqrt(23)/2


# ## Define potential and strain from QNM

# potchel-teller potential
def vx(x):
    lambdari=lambdar+1j*lambdai;
    return -lambdari*(lambdari+1)/(cosh(x))**2

# contribution to strain from n-th qnm
def term_n(t,n):
    temp=-1/2*((-1)**n)*gamma(n+1)*exp((-(n+1/2)+1j*lambdai)*t)*gamma(-n+2*1j*lambdai)/ ((-(n+1/2)+1j*lambdai)*(gamma(-(n+1/2)+1j*lambdai))**2);
    return temp + conjugate(temp)

# total strain
def psi(t):
    total=0;
    for k in range(0,4):
        total = total + term_n(t,k);
        return total
    


# ## Plot strain-time

ts=np.arange(10,30,0.1)
term0=term_n(ts,0)
term1=term_n(ts,1)
term2=term_n(ts,2)
term3=term_n(ts,3)
psis=psi(ts)

fig6=plt.figure(figsize=(6,6))
plt.plot(ts,arcsinh(100000000000000*term0),label='n=0',color='red')
plt.plot(ts,arcsinh(100000000000000*term1),label='n=1',color='blue')
plt.plot(ts,arcsinh(100000000000000*term2),label='n=2',color='green')
plt.legend()
plt.xticks(([10,15,20,25,30]))
plt.xlabel('t')
plt.ylabel('$\sinh^{-1}(10^{14}\psi(t,x=5))$')
fig6.savefig("h-extrembh.pdf",facecolor='white')


# ## Plot potchl-teller potential

fig7=plt.figure(figsize=(6,6))
xs=np.arange(-7,7,0.1)
pot=plt.plot(xs,vx(xs),label='Potential',color='black')
vline=plt.axvline(-5,0.05,0.6,label='Initial data',color='red')
arrow=plt.arrow(5,1.18,0,-1,width=0.05,label='Detector',color='blue')
plt.xticks(([-5,0,5]))
#plt.legend([vline,pot,arrow], ['Initial data','Potential','Detector'])
plt.text(-6,3.9,'Initial data')
plt.text(-1,1.5,'Potential')
plt.text(4,1.3,'Detector')
plt.xlabel('x');
fig7.savefig("v-extrembh.pdf",facecolor='white')
