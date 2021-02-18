#!/usr/bin/env python
# coding: utf-8

# # Black hole quasinormal ringing in dS_Sch or MG black holes


# Import libraries
import math
import numpy as np
import pandas as pd
from numpy import sqrt,pi,exp,conjugate,real,arcsinh,cosh
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
from scipy.optimize import curve_fit


# Check the table of outputs from solving Laplace equation
schro.head(3)


# Define source and fitting function

qnm = [0.747, 0.178];  #Kokkotas 1999

def strain_old(t,schro1):
    cosine = np.cos(t*schro1.w);
    sine = np.sin(t*schro1.w);
    dw = schro1.w[2] - schro1.w[1];
    return dw*(np.dot(schro1.PsiR, cosine) + np.dot(schro1.PsiI, sine))

def strain_gaussian(t,schro1,x0=-10,sigma=1):
    term_const = sigma/4/sqrt(pi);
    term_Ain = 1/(schro1.CinR + 1j*schro1.CinI);
    term_exp = exp(-1j*t*schro1.w- sigma**2/2*((schro1.w)**2));
    dw = schro1.w[2] - schro1.w[1];
    return dw*term_const*np.dot(term_Ain, term_exp)
    
def strain_wminmax(t,schro1,wmin=0,wmax=3,sigma=4):
    schro2=schro1.loc[schro1['w']>=wmin][schro1['w']<=wmax];
    term_const = sigma/4/sqrt(pi);
    term_Ain = 1/(schro2.CinR + 1j*schro2.CinI);
    term_exp = exp(-1j*t*schro2.w - sigma**2/2*((schro2.w)**2));
    dw = schro2.w[2] - schro2.w[1];
    return dw*term_const*np.dot(term_Ain, term_exp)

def strain(t,schro1):
    return strain_gaussian(t,schro1)

def mse(y_pred, y_data):
    return sum(np.power(y_pred-y_data,2))/len(y_pred)

def func(x, wr,wi,A,phi):
    return A*np.cos(wr*x+phi)*np.exp(-wi*x)

def best_t(t,h,dt,toplot=1):
    error = 10;
    tpart = np.zeros(dt);
    hpart = np.zeros(dt);
    for ii in np.arange(0,len(t)-dt):
        t1 = t[range(ii,ii+dt)]
        h1 = h[range(ii,ii+dt)];
        popt, pcov = curve_fit(func, t1-t1[0], h1);
        error1 = mse(func(t1-t1[0],*popt),h1);
        if error1<error:
            error = error1;
            tpart = t1;
            hpart = h1;
    popt, pcov = curve_fit(func, tpart-tpart[0], hpart,bounds=([0,0,-np.inf,0],+np.inf));
    rel_para = (popt[[0,1]] - qnm)/qnm;
    if toplot:
        plt.plot(tpart,func(tpart-tpart[0],*popt))
        plt.plot(tpart,hpart)
    return rel_para,popt[[0,1]],[tpart[0],tpart[-1]],error

def get_schro(file):
    schro=pd.read_csv(file,sep='\t', header=None)
    schro.rename(columns={0:'w',1:'CinR',2:'CinI',3:'PsiR',4:"PsiI",5:'P'},inplace=True);
    return schro

def get_pow(schro):
    w=schro.w;
    p=schro.P;
    return w,p

def get_ht(schro,t0,te,dt=0.2):
    t=np.arange(t0,te,dt);
    h = np.zeros(len(t));
    for j in range(len(t)):
        h[j] = strain(t[j],schro);
    return t,h

def get_vr(file):
    vr=pd.read_csv(file,sep='\t', header=None)
    vr.rename(columns={0:'rt',1:'v'},inplace=True);
    return vr


# Plot strain for Rd=10, B=0

schro=get_schro('./data/schrodinger_rd10_0.txt');
t0,h0 = get_ht(schro,-10,200);
t,h = get_ht(schro,10,30);

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot(t,np.log(np.abs(h)))
plt.subplot(1, 2, 2)
plt.plot(t0,h0)

best_t(t,h,50)


# Plot strain for R=10, Bb=0,3,5,10, t0 adjusted by setting t0=0 at GW source

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
schro=get_schro('./data/schrodinger_rd10_0.txt');
t0,h0 = get_ht(schro,-10,200);
plt.plot(t0,h0)
plt.subplot(2,2,2)
schro=get_schro('./data/schrodinger_rd10_3.txt');
t0,h0 = get_ht(schro,-10,200);
plt.plot(t0,h0)
plt.subplot(2,2,3)
schro=get_schro('./data/schrodinger_rd10_5.txt');
t0,h0 = get_ht(schro,-10,200);
plt.plot(t0,h0)
plt.subplot(2,2,4)
schro=get_schro('./data/schrodinger_rd10_10.txt');
t0,h0 = get_ht(schro,-10,200);
plt.plot(t0,h0)


# Plot strain R=10, Bb=0,3,10 for t=[0,10]

t0_plot=0;
te_plot=100;
fig1=plt.figure(figsize=(6,6))
schro=get_schro('./data/schrodinger_rd10_0.txt');
t0,h0 = get_ht(schro,t0_plot,te_plot);
plt.plot(t0+60,h0,color='black',linestyle='solid',label='B=0')

schro=get_schro('./data/schrodinger_rd10_3.txt');
t0,h0 = get_ht(schro,t0_plot,te_plot);
plt.plot(t0+60,h0,color='red',linestyle='dotted',label='B=3')

schro=get_schro('./data/schrodinger_rd10_10.txt');
t0,h0 = get_ht(schro,t0_plot,te_plot);
plt.plot(t0+60,h0,color='blue',linestyle='dashed',label='B=10')

plt.legend()
plt.xlabel('t')
plt.ylabel('$\psi(t,r_*=60)$')
ax=plt.gca()
ax.yaxis.set_label_coords(-0.1, 0.6)


# presentation plot
t0_plot=0;
te_plot=100;
fig1=plt.figure(figsize=(6,6))


schro=get_schro('./data/schrodinger_rd10_0.txt');
t0,h0 = get_ht(schro,t0_plot,te_plot);
plt.plot(t0+60,h0,color='black',linestyle='solid',label='B=0')
#plt.legend(['B=0'],prop={'size':80}) #,'B=3','B=10']

schro=get_schro('./data/schrodinger_rd10_3.txt');
t0,h0 = get_ht(schro,t0_plot,te_plot);
plt.plot(t0+60,h0,color='red',linestyle='dotted',label='B=3')

schro=get_schro('./data/schrodinger_rd10_10.txt');
t0,h0 = get_ht(schro,t0_plot,te_plot);
plt.plot(t0+60,h0,color='blue',linestyle='dashed',label='B=10')
plt.legend(loc=2, prop={'size': 20})
ax = plt.gca()
ax.set_facecolor('xkcd:white')
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)

plt.legend()
plt.xlabel('t',fontsize=20)
plt.ylabel('$\psi(t,r_*=60)$',fontsize=20)
ax=plt.gca()
ax.yaxis.set_label_coords(-0.1, 0.6)


fig1.savefig("strain.pdf",facecolor='white')


# Plot power spectrum for R=10, B=0,3,10

fig4=plt.figure(figsize=(6,6))
schro=get_schro('./data/schrodinger_rd10_0.txt');
w,p = get_pow(schro);
plt.plot(w,p,color='black',linestyle='solid',label='B=0')

schro=get_schro('./data/schrodinger_rd10_3.txt');
w,p = get_pow(schro);
plt.plot(w,p,color='red',linestyle='dotted',label='B=3')

schro=get_schro('./data/schrodinger_rd10_10.txt');
w,p = get_pow(schro);
plt.plot(w,p,color='blue',linestyle='dashed',label='B=10')

plt.legend()
plt.xlim(0.3,3)
plt.xlabel('$\omega$')
plt.ylabel('P')
fig4.savefig("pow.pdf",facecolor='white')


# ## For different rd and B


# Fit principal QNM for different R and B

rds=[20,40,80,160];
Bs=[1,2,3,4,5,6];
rel_paraR=np.zeros([len(rds),len(Bs)]);
rel_paraI=np.zeros([len(rds),len(Bs)]);
for k in range(len(rds)):
    for j in range(len(Bs)):
        rd=rds[k];
        B=Bs[j];
        schro=get_schro('./data/schrodinger_rd'+str(rd)+'_D'+str(B)+'.txt');
        t,h = get_ht(schro,10,30);
        rel_para,_,_,_=best_t(t,h,50,toplot=0);
        rel_paraR[k][j]=rel_para[0];
        rel_paraI[k][j]=rel_para[1];

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
for j in range(len(Bs)):
    plt.plot(rds,rel_paraR[:,j],label=str(Bs[j])) #np.log(np.abs(rel_paraR[:,j]))
plt.legend()

plt.subplot(1,2,2)
for k in range(len(rds)):
    plt.plot(Bs,rel_paraR[k,:],label=str(rds[k])) #np.log(np.abs(rel_paraR[k,:]))
plt.legend()


# ## for rd=20 only


# Fit principal QNM for R=10 and different Bs

rds=10;
Bs=['0','D1','D2','D3','D4','D5','D6','D7','D8','D9','1','1D1','1D3','1D5','1D7','1D9','2','2D1','2D3','2D5','2D7','2D9','3','3D1','3D3','3D5','3D7','3D9','4'];#,'1D1','1D2','1D3','1D4','1D5','1D6','1D7','1D8','1D9','2'];#,'3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'];
rel_paraR=np.zeros(len(Bs));
rel_paraI=np.zeros(len(Bs));
for j in range(len(Bs)):
    rd=rds;
    B=Bs[j];
    schro=get_schro('./data/schrodinger_rd'+str(rd)+'_'+B+'.txt');
    t,h = get_ht(schro,10,30);
    rel_para,_,_,_=best_t(t,h,50,toplot=0);
    rel_paraR[j]=rel_para[0];
    rel_paraI[j]=rel_para[1];

# Plot the difference of QNM relative to Schwarzschild vs B

Bnum=np.concatenate((np.arange(0,1.1,0.1),np.arange(1.1,2.1,0.2),np.array([2,2.1,2.3,2.5,2.7,2.9,3,3.1,3.3,3.5,3.7,3.9,4])));
#Bnum=np.array([0,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]);
#Bnum=np.concatenate((np.arange(0,2.1,0.1), np.arange(3,13)));
fig=plt.figure(figsize=(6,6))
plt.plot(Bnum,100*rel_paraR,label='real part')
plt.plot(Bnum,100*rel_paraI,label='imaginary part')
plt.legend()
plt.xlabel('B')
plt.ylabel('% difference from Schwarzschild QNM')
plt.xlim(0,3.5)
plt.ylim(-20,40)
fig.savefig("rd10.pdf",facecolor='white')

np.concatenate((np.arange(0,1.1,0.1),np.arange(1.1,2.1,0.2),np.array([2,2.1,2.3])))


# Plot the effective potential

fig3=plt.figure(figsize=(6,6))
vr=get_vr('./data/vrlist_rd10_B0.txt')
vr.sort_values('rt',inplace=True)
plt.plot(vr.rt,vr.v,linestyle='solid',color='black',label='B=0')
vr=get_vr('./data/vrlist_rd10_B3.txt')
vr.sort_values('rt',inplace=True)
plt.plot(vr.rt,vr.v,linestyle='dotted',color='red',label='B=3')
vr=get_vr('./data/vrlist_rd10_B10.txt')
vr.sort_values('rt',inplace=True)
plt.plot(vr.rt.values,vr.v.values,linestyle='dashed',color='blue',label='B=10')

r_init=np.arange(-15,-7,0.1);
h_init=np.exp(-(r_init+10)**2)
plt.plot(r_init,h_init,linestyle='solid',color='green',label='Initial data')
plt.arrow(50,0.15,0,-0.15,width=0.04,head_width=1,length_includes_head=True,        head_length=0.03,label='Detector',color='magenta')
plt.text(45,0.16,'Detector')
plt.xlim(-15,55)
plt.xlabel(r'$r_*$')
plt.ylabel('V')
plt.legend()
fig3.savefig("Vr.pdf",facecolor='white')


