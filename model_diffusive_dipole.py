# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:36:20 2019

@author: Akash
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from scipy import ndimage
from scipy import signal
from scipy.signal import convolve2d
from scipy.optimize import minimize
from scipy.optimize import least_squares
from numpy.linalg import inv
from scipy.sparse import vstack

#initialize global variables
#beta = 1.2
mua = 0.0458 #0.005
musp = 3.56541
g = 0.9

Nr = 100
Nc = 100
Nz = 100


Reff = 0 
beta = math.sqrt(3*musp*mua)
# Ground truth vein structure

Vn = np.ones(shape=(Nr,Nc,Nz))

# Kernel size

# Multiplication factor 1 px = mf mm
mf   = .1 #.005
mf_z = 1 #.005

def convolve3D(arr1,arr2):
    Z = arr1.shape[2]
    I = []
    arr2 = np.flip(np.flip(arr2,0),1)
    for d in range(0,Z):
        im_convolv = convolve2d(arr1[:,:,d], arr2[:,:,d], mode="same", boundary="symm")
        I.append(im_convolv)
    Image2d = np.sum(np.array(I),axis=0)
    return Image2d


def getW(td,n): 
    # Phase function for every td
    Gs = getKernel(td,n)
    Gs = np.reshape(Gs,(Nz,n,n))
    a = np.arange(Nr*Nc).reshape(Nr,Nc)
    vect_ind = np.zeros(shape=())
    
    sub_shape = (n,n)
    view_shape = tuple(np.subtract(a.shape, sub_shape) + 1) + sub_shape
    strides = a.strides + a.strides
    
    sub_matrices = np.lib.stride_tricks.as_strided(a,view_shape,strides)
    count = 0
    p,q,r,s = sub_matrices.shape
    vect_ind = np.zeros(shape=(p*q,r*s))
    for i in range(0,p):
        for j in range(0,q):
            vect_ind[count,:] = np.reshape(sub_matrices[i,j,:,:],(1,r*s))
            count = count+1
    
    vect_ind = vect_ind.astype(np.int)
    g = np.reshape(Gs,(n*n,Nz))
    
    W = np.zeros(shape=(p*q,Nr*Nc*Nz))
    
    for z in range(0,Nz):        
        for row in range(0,p*q):
            count = 0
            for i in range(0,r*s):
                W[row,(z*Nr*Nc)+vect_ind[row,i]] = g[count,z]
                count = count + 1
                
    return W
                
            
            
def getKernel(td,n): 
    # Phase function for every td
    P = np.zeros(shape=(n,n,Nz,n))
    z0 = 1/(musp+mua);
    D = 1/(3*(mua+musp));
    zb = (1+Reff)/(1-Reff)*2*D;
    Gs = np.zeros(shape=(n,n,Nz))
    s = n/2* mf
    t = n/2* mf
    up = n/2 - td
    u = up*mf

    for vp in range(0,n):
        v = vp * mf
        for rxp in range(0,n):
            rx = rxp*mf
            for ryp in range(0,n):
                ry=ryp*mf
                for rzp in range(1,Nz):
                    rz= rzp * mf * mf_z

                    rs1 = np.sqrt((u-rx)**2+(v-ry)**2+(0+z0-rz)**2)
                    rs2 = np.sqrt((u-rx)**2+(v-ry)**2+(0-z0-2*zb-rz)**2)

                    phi_1 = math.exp(-beta*rs1)/rs1-math.exp(-beta*rs2)/rs2

                    # dipole for G(r, r_d)
                    rd1 = np.sqrt((rx-s)**2+(ry-t)**2+ ( 0+z0-rz)**2)
                    rd2 = np.sqrt((rx-s)**2+(ry-t)**2+ ( 0-z0-2*zb-rz)**2)
                    phi_2 = math.exp(-beta*rd1)/rd1-math.exp(-beta*rd2)/rd2

                    num = phi_1*phi_2

                    # denom = \phi(rs, rd)
                    rsd1 = np.sqrt((u-s)**2+(v-t)**2+(0+z0-0)**2)
                    rsd2 = np.sqrt((u-s)**2+(v-t)**2+(0-z0-2*zb-0)**2)
                    phi_0 = (math.exp(-beta*rsd1)/rsd1-math.exp(-beta*rsd2)/rsd2)
                    P[rxp,ryp,rzp,vp] = num/phi_0

    Gs = np.sum(P,axis=3)
#    plt.imshow(Gs[:,:,1])
#    plt.show()
    return Gs

    
def getIntensity(td,n):
    K0=[]    
    s = n/2*mf
    t = n/2*mf
    up = n/2-td
    u = up*mf
    for vp in range(0,n):
        v = vp * mf
        z0 = 1/(musp+mua);
        D = 1/(3*(mua+musp));
        zb = (1+Reff)/(1-Reff)*2*D;
        rsd1 = np.sqrt((u-s)**2+(v-t)**2+(0+z0-0)**2)
        rsd2 = np.sqrt((u-s)**2+(v-t)**2+(0-z0-2*zb-0)**2)
        phi_0 = (math.exp(-beta*rsd1)/rsd1-math.exp(-beta*rsd2)/rsd2)
        K0.append(phi_0)
    I0 = 7*(np.sum(np.array(K0)))/(math.pi*4)
    return I0

def obj_fcn(qx, Im,Gs,In):
    q = np.reshape(qx,(Nr,Nc,Nz))
    I = convolve3D(q,Gs)
    I_fwd = 5*np.divide(I,In)
    im1 = np.reshape(I_fwd, -1, 1)
    im2 = np.reshape(Im, -1, 1)
    err = np.sum(np.power(im1-im2, 2))/np.sum(np.power(im2, 2))
    return err
    
   
def main():
    pass
#    res = inverse(Image,Gs,In)
#    q_opt = res.x
#    q_opt = q_opt.reshape(Nr,Nc,Nz)
    
#    ####### Invrese modelling/ Optimization 
#    ## Stack measurement into a vector
#    TD = 30
#    Im = np.zeros(shape=(TD-5,25,25))
#    count=0
#    for t in range(0,TD):
#        td = t - TD/2
#        if(abs(td)<3):
#            continue
#        W = getW(td,n)
#        Vq = np.reshape(V,Nr*Nc*Nz)
#        I_hat = W.dot(Vq)
#        Im[count,:,:] = np.reshape(I_hat,(25,25))
#        count=count+1
#    
#    ## Stack the design matrix 
#    W_prime=[]
#    I_meas=[]
#    count = 0
#    Vq = np.reshape(V,Nr*Nc*Nz)
#    for t in range(0,TD):
#        td = t - TD/2
#        if(abs(td)<3):
#            continue
#        W = getW(td,n)
#        W_prime.append(W) 
#        I_hat = np.reshape(Im[count,:,:],25*25)
#        I_meas.append(I_hat)
#        count=count+1
#    W_prime = np.concatenate(W_prime,axis=0)
#    I_meas = np.concatenate(I_meas,axis=0)
#    ## ridge regression
#    lamda = 1000
#    order = W.shape[1]
#    q_hat = inv(W_prime.T.dot(W_prime)+lamda*np.identity(order)).dot(W_prime.T).dot(I_meas)
#    Q_hat = np.reshape(q_hat,(Nz,Nr,Nc))
#    plt.plot(Q_hat[:,17,16])
    

    
    

  
def inverse(Im,Gs,In):
    q0 = np.zeros(shape=(Nr,Nc,Nz))
    q0 = np. reshape(q0,Nr*Nc*Nz)
    
    res = minimize(obj_fcn, q0, args=(Im,Gs,In),method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    return res

          
if __name__ == '__main__':
    main()
    from matplotlib.pyplot import *
    TD = 10
    count = 0
    vein_r = 3
    n = 32
    N = 1
    mat = np.zeros(shape=(N,20))
    ####### Forwardr modelling/ Generate images
    for tv in np.linspace(0,15,N):
        tv = 10
        dv = int(tv)
        print(dv)
        Intensity_vein=[]
        Intensity_back=[]
        for t in range( 0, TD ):
            td = t - TD/2
            Gs = getKernel(td,n)
            Vcon = np.zeros(shape=(Nr,Nc,Nz))
            Vn = np.ones(shape=(Nr,Nc,Nz)) * 1

            Vcon[ 50 - vein_r : 50 + vein_r + 1 , : , dv - vein_r :dv + vein_r + 1] = 1
            # Vcon[ 31:33 , : , dv + 1 :dv + 3] = 1

            I = convolve3D(Vcon,Gs)
            In = convolve3D(Vn,Gs)

            I0 = 800*getIntensity(td,n)
            # Image = I0*(1-100*np.divide(I,In))
            Image = I0*(1 - np.divide(I,In))
            Image_norm = 1 - np.divide(I,In)

            import ipdb
            ipdb.set_trace()
            print(); 
    




