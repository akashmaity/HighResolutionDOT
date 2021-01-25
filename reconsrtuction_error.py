#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:12:09 2019

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

def crop_center(img,cropx,cropy):
    z,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:,starty:starty+cropy,startx:startx+cropx]

dv = 1.5
mf = 0.25

mat = scipy.io.loadmat('green_Q_opt_d_%.1f_tdmin_3_Bonn.mat'%dv)
Q_opt1 = mat['Q_opt']
IMGS_gt = mat['imgs_gt']
IMGS_opt = mat['imgs_opt']
IMGS_ref = mat['Imgs']

#dv = 3.5
#mat = scipy.io.loadmat('Q_opt_d_%.1f_tdmin_5_Bonn.mat'%dv)
#Q_opt2 = mat['Q_opt']
##
z = np.arange(0,32)*mf
#
Q_gt = np.zeros(32)
Q_gt[int(dv/mf-1):int(dv/mf+1)]=0.170
#
#plt.plot(z,Q_opt1[:,50,50]);plt.plot(z,Q_gt);#plt.plot(z,Q_opt1[:,48,60])#;
#plt.xlabel('Depth')
#plt.legend(('tdmin2','tdmin5','GT'))
#plt.savefig('d%.1f.jpg'%dv)
##
#img = np.hstack([Q_opt1[4,:,:], Q_opt1[9,:,:]])
#plt.imshow(img)
TD = 40 - 10
img = np.hstack([IMGS_opt[TD,:,:], IMGS_ref[TD,:,:], IMGS_gt[TD,:,:]])
plt.imshow(img)
#plt.imshow(Q_opt1[5,:,:])

#max_num = np.max(Q_opt)
#Q_opt = Q_opt/max_num
#Q_opt[Q_opt<1]=0
#Q_opt = np.nan_to_num(Q_opt)
#Q_opt = crop_center(Q_opt,69,69)

#Nz = 32
#Nr = 100
#Nc = 100
#Q_gt = mat['Q_gt']
#Q_gt[Q_gt>0] = 1
#Q_gt = np.zeros(shape=(Nz, Nr, Nc))
#Q_gt[ 6-3: 6+3, :, 50-3:53 ] = 2.3 
#Q_gt[ 6-3: 6+3, 50-3:53, : ] = 2.3 

#max_num = np.max(Q_gt)
#Q_gt = Q_gt/max_num
#Q_gt = np.nan_to_num(Q_gt)
#Q_gt = crop_center(Q_gt,69,69)

#Error = np.linalg.norm(Q_gt - Q_opt)/np.linalg.norm(Q_gt)

#plt.plot(z,Q_opt1[:,48,50]);plt.plot(z,Q_gt)
#print(np.argmax(Q_opt1[:,48,50])*0.25)
#
##plt.plot(Q_opt[np.argmax(Q_opt[:,50,50]),:,50])
#
#plt.plot(Q_opt[:,48,50])
#plt.imshow(Q_opt[12,:,:])
#img = np.hstack([Q_opt2[5,:,:], Q_opt2[2,:,:]])
#plt.imshow(img)

#cmap = plt.cm.OrRd
#cmap.set_under(color='black')  
#
#for i in range(0,32):
#    img = np.hstack( ( Q_gt[i,:,:], Q_opt[i,:,:]) )
##    img = Q_opt[i,:,:]
#    plt.imshow(img, cmap = cmap, vmin = 0.0001, vmax = 5)
#    plt.savefig('dat/video/d%d.png'%i)
##    plt.colorbar()
#    plt.draw()
#    plt.show()

#cmap = plt.cm.OrRd
#cmap.set_under(color='black')    
#
#mat1 = scipy.io.loadmat('Image_dTs_mf_0.2.mat')
#mat2 = scipy.io.loadmat('Image_dTs_mf_1.mat')
#
#IMGS_1 = mat1['IMGS_ref']
#IMGS_2 = mat2['IMGS_ref']
#
#TD = 29
#img = np.hstack([IMGS_1[TD,:,:], IMGS_2[TD,:,:]])
#plt.imshow(img)


