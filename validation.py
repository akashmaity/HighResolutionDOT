#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:45:09 2019

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
import scipy.io as sio

#mat = sio.loadmat('/home/akash/Desktop/MMC_simulations/epi_mcx-master/intensity_dipole_opt.mat')
#Id = mat['Id']

#mat= sio.loadmat('Q_opt_d_8_tdmin_5.mat') # image td dipole model, moving detector, fix source; fixed detector, moving source
#IMGS_ref = mat['imgs_opt']
#IMGS_ref = np.transpose(IMGS_ref,(1,2,0))
#IMGS_td = mat['Imgs']
#IMGS_td = np.transpose(IMGS_td,(1,2,0))
#
#
##mat2= sio.loadmat('Image_dTs_10.mat') # image td dipole model, moving detector, fix source; fixed detector, moving source
#IMGS_ref2 = mat['imgs_gt']
#IMGS_ref2 = np.transpose(IMGS_ref2,(1,2,0))
#IMGS_ref[IMGS_td==0]=0
#IMGS_ref2[IMGS_td==0]=0

mat = sio.loadmat('Q_opt_d_6.0_tdmin_4_Rytov_33.mat')
IMGS_opt = mat['imgs_opt']
IMGS_opt = np.transpose(IMGS_opt,(1,2,0))

IMGS_gt = mat['imgs_gt']
IMGS_gt = np.transpose(IMGS_gt,(1,2,0))

IMGS_td = mat['Imgs']
IMGS_td = np.transpose(IMGS_td,(1,2,0))

mat = sio.loadmat('./dat/img_dt_dvein_6.0_rvein_1.mat')
IMGS_ref2 = mat['IMGS_td']
#IMGS_ref = np.transpose(IMGS_ref,(1,2,0))

#mat = sio.loadmat('./dat/img_dt_dvein_4.0_rvein_1.mat')
#IMGS_td = mat['IMGS_td']
#IMGS_td = np.transpose(IMGS_td,(1,2,0))


#src_row = 20
#intensity = []
#for d in range(21,61):
#    det_row = d
#    td = det_row - src_row
#    intensity.append(IMGS_ref[det_row,50,40-td])
#plt.plot(np.log10(intensity)); plt.plot(np.squeeze(Id))
#plt.show()

TD = 40 + 25

plt.plot(np.log10(IMGS_opt[:,50,TD]));plt.plot(np.log10(IMGS_td[:,50,TD]));plt.plot(np.log10(IMGS_gt[:,50,TD]));plt.plot(np.log10(IMGS_ref2[:,50,TD]))
###
#I1=[]
#I2=[]
for td in np.arange(-40,41):
    t=td+40
    print (td)
#    I1.append(np.log10(np.linalg.norm(IMGS_opt[:,:,t]-IMGS_td[:,:,t])))
#    I2.append(np.log10(np.linalg.norm(IMGS_gt[:,:,t]-IMGS_td[:,:,t])))
    IMGS_ref[IMGS_td==0]=0
    IMGS_opt[IMGS_td==0]=0
    print(np.linalg.norm(IMGS_ref[:,20:80,TD]-IMGS_td[:,20:80,TD]))#/np.linalg.norm(IMGS_td))
    print(np.linalg.norm(IMGS_ref2[:,20:80,TD]-IMGS_td[:,20:80,TD]))#/np.linalg.norm(IMGS_td))
###
#plt.plot(I1);plt.plot(I2)
###plt.show()
img = np.hstack([IMGS_opt[:,:,TD], IMGS_td[:,:,TD],  IMGS_gt[:,:,TD], IMGS_ref2[:,:,TD]])
plt.imshow(img)

#sio.savemat('/home/akash/Desktop/MMC_simulations/epi_mcx-master/intensity_dipole_td.mat',mdict={'Id': np.log10(intensity)})
