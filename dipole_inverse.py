# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:24:59 2019

@author: Akash
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:53:50 2019

@author: Akash
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:36:20 2019

@author: Akash
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 08:32:34 2019

@author: Akash
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:53:56 2019

@author: akash
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
import scipy.io 
from scipy import sparse
from numpy import array
from sklearn.linear_model import Lasso

#initialize global variables
#beta = 1.2
g = 0.9
Nr = 100
Nc =  100
Nz = 32
K = 10**11
mua = 0.045
musp = 35.65*(1-g)
Reff = 0 
D = 1/(3*(mua+musp))
beta = math.sqrt(3*musp*mua)
#beta = 0.69
# Ground truth vein structure
dv= 6
rv = 4
#        Vshape = np.zeros(shape=(Nz,Nr,Nc))
#        Vshape[dv-rv:dv+rv,int(Nr/2-rv):int(Nr/2+rv),:] = 10

Vn = np.ones(shape=(Nr,Nc,Nz))

# Kernel size

# Multiplication factor 1 px = mf mm
mf = 0.1

def convolve3D(arr1,arr2):
    K = 10**8
    Z = arr1.shape[2]
    I = []
    arr2 = np.flip(np.flip(arr2,0),1)
    for d in range(0,Z):
        im_convolv = convolve2d(arr1[:,:,d], arr2[:,:,d], mode="same", boundary="symm")
        I.append(im_convolv)
    Image2d = np.sum(np.array(I),axis=0)
    return (Image2d*K)


def getW_sparse(td,n):
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
    
    indices_col = []
    indices_row = []
    values = []
    for row in range(0,p*q):
        for z in range(0,Nz):        
            count = 0
            for i in range(0,r*s):
                indices_col.append((z*Nr*Nc)+vect_ind[row,i])
                indices_row.append(row)
                values.append(g[count,z])
                count = count + 1
    W = sparse.coo_matrix((np.array(values),(np.array(indices_row),np.array(indices_col))),shape=(p*q,Nr*Nc*Nz)).tocsr()
    return W
                
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
            
            
def getKernel_monopole(td,n): 
    # Phase function for every td
    P = np.zeros(shape=(n,n,Nz,n))
    Gs = np.zeros(shape=(n,n,Nz))
    s = n/2*mf
    t = n/2*mf
    up = n/2-td
    u = up*mf
    for vp in range(0,n):
        v = vp * mf
        for rxp in range(0,n):
            rx = rxp*mf
            for ryp in range(0,n):
                ry=ryp*mf
                for rzp in range(1,Nz):
                    rz=rzp*mf
                    num = math.exp(-beta*(np.sqrt((rx-u)**2+(ry-v)**2+rz**2)+np.sqrt((rx-s)**2+(ry-t)**2+rz**2)-np.sqrt((s-u)**2+(t-v)**2)))
                    deno = np.sqrt((rx-u)**2+(ry-v)**2+rz**2)*np.sqrt((rx-s)**2+(ry-t)**2+rz**2)/np.sqrt((s-u)**2+(t-v)**2)
                    P[rxp,ryp,rzp,vp] = num/deno
    Gs = np.average(P,axis=3)
#    plt.imshow(Gs[:,:,1])
#    plt.show()
    return Gs

def getKernel(td,n):
    # Phase function for every td
    P = np.zeros(shape=(n,n,Nz,n))
    z0 = 1/(musp+mua);
    zb = (1+Reff)/(1-Reff)*2*D;
    Gs = np.zeros(shape=(n,n,Nz))
    s = n/2*mf
    t = n/2*mf
    up = n/2-td
    u = up*mf
    for vp in range(0,n):
        v = vp * mf
        for rxp in range(0,n):
            rx = rxp*mf
            for ryp in range(0,n):
                ry=ryp*mf
                for rzp in range(1,Nz):
                    rz=rzp*mf
                    rs1 = np.sqrt((u-rx)**2+(v-ry)**2+(0+z0-rz)**2)
                    rs2 = np.sqrt((u-rx)**2+(v-ry)**2+(0-z0-2*zb-rz)**2)
                    phi_1 = (math.exp(-beta*rs1)/rs1-math.exp(-beta*rs2)/rs2)
                    rd1 = np.sqrt((rx-s)**2+(ry-t)**2+(0+z0-rz)**2)
                    rd2 = np.sqrt((rx-s)**2+(ry-t)**2+(0-z0-2*zb-rz)**2)
                    phi_2 = (math.exp(-beta*rd1)/rd1-math.exp(-beta*rd2)/rd2)
                    num = phi_1*phi_2

                    rsd1 = np.sqrt((u-s)**2+(v-t)**2+(0+z0-0)**2)
                    rsd2 = np.sqrt((u-s)**2+(v-t)**2+(0-z0-2*zb-0)**2)
                    phi_0 = (math.exp(-beta*rsd1)/rsd1-math.exp(-beta*rsd2)/rsd2)
                    P[rxp,ryp,rzp,vp] = num/phi_0

    Gs = np.average(P,axis=3)/(4*math.pi*D)
    Gs = Gs/Gs.sum()
    return Gs

    
def getIntensity(td):
    K0=[]    
    t = Nc/2*mf
    for vp in range(0,Nc):
        v = vp * mf
        z0 = 1/(musp+mua);
        zb = (1+Reff)/(1-Reff)*2*D;
        rsd1 = np.sqrt((td*mf)**2+(v-t)**2+(0+z0-0)**2)
        rsd2 = np.sqrt((td*mf)**2+(v-t)**2+(0-z0-2*zb-0)**2)
        phi_0 = 1/(4*math.pi*D)*(math.exp(-beta*rsd1)/rsd1-math.exp(-beta*rsd2)/rsd2)
        K0.append(phi_0)
    I0 = (np.sum(np.array(K0)))
    return I0

    
   
def main():
    pass


          
if __name__ == '__main__':
    main()
    n = Nz
    Id=[]
    W_prime=[]
    I_meas=[]
    count = 0
    # mat = scipy.io.loadmat('/home/akash/Desktop/Diffuse_model/img_dt_dvein_6_rvein_1.mat')
    mat = scipy.io.loadmat('./dat/dipole_validate/img_dt_dvein_6_rvein_1.mat')
    Images = mat['IMGS_td']
    dT = mat['dTs'].flatten()
    TD = 120 

    # Optimizing ||Y - W * Q||2 + other constraints
    for td in range(0,TD,1):
        print(td)
        t = td - TD/2 

        if(abs(t)<3):
            continue 

        # Creating W for each images
        W = getW_sparse(-t,n)
        W_prime.append(W) 
        # Creating measurement vector Y
        # Y = (1 - Im/I0) by Born approximation
        # The images are cropped due to boundary effects arising from the convolution
        I0 = getIntensity(-t)
        I_m = crop_center(Images[:,:,td],69,69)
        I_m = 1-np.reshape(I_m,69*69)/I0 ## Born
#        I_hat = -np.log(np.reshape(I_hat,69*69)/np.max(I_hat)) ## Rytov
        I_meas.append(I_m)                    

    W_prime = vstack(W_prime)
    # Saving the W matrix so need not compute everytime
    scipy.sparse.save_npz('/home/akash/Desktop/Diffuse_model/W_prime_120.npz',W_prime)
    W_prime = scipy.sparse.load_npz('/home/akash/Desktop/Diffuse_model/W_prime_120.npz')
        
    I_meas = np.concatenate(I_meas,axis=0)
    
    
    # Reconstructing Q by Lasso formulaion with non-negative constraint
    clf = Lasso(alpha=0.001,precompute=True,max_iter=1000,positive=True, random_state=9999, selection='random')
    clf.fit(W_prime, I_meas)
    q_hat = clf.coef_
    Q_hat = np.reshape(q_hat,(Nz,Nr,Nc))
    plt.plot(Q_hat[:,50,50])
      
    # Ground truth vein structure
#    V = np.zeros(shape=(Nz,Nr,Nc))
#    V[7:9,16:18,:] = 1
#    Vq = np.reshape(V,Nr*Nc*Nz)
#    I_meas = W_prime.dot(Vq)
#    clf = Lasso(alpha=0.01,precompute=True,max_iter=1000,positive=True, random_state=9999, selection='random')
#    clf.fit(W_prime, I_meas)
#    q_hat = clf.coef_
#    Q_hat = np.reshape(q_hat,(Nz,Nr,Nc))
#    plt.plot(Q_hat[:,17,16])
#    
        

    


    

                
                        

    
                

    

 



  


