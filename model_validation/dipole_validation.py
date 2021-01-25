from  matplotlib.pyplot import *
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

import torch

#initialize global variables
#beta = 1.2
Nr = 100
Nc = 100
Nz = 33


# optical properties for the homogenous volume #
mua = 0.0458/4
g = .9
musp = 35.65*(1-g)
musp = 10*(1-g)
# debug:
#mua = .005
#g = 0.01 #0.9
#musp = 1 * (1-g)

Reff = 0 
D = 1/(3*(mua+musp))
beta = math.sqrt(3*musp*mua)
is_slit = 1

dv = 6 # depth of vein

# optical properties for the inhomogenous volume #
Vcon = np.zeros(shape=(Nr,Nc,Nz))
#debug:
#Vcon[50-3:50+3,:,dv-3:dv+3] = 0
#
Vcon[50-3:50+3,:,dv-3:dv+3] = 2

# Multiplication factor 1 px = mf mm
mf = 1 #0.1 
          
def getKernel(td,n):
    # Phase function for every td = u -s 
    z0 = 1/(musp+mua);
    zb = (1+Reff)/(1-Reff)*2*D;
    Gs = np.zeros(shape=(n,n,Nz))

    s = n/2 * mf
    t = n/2 * mf
    up = n/2-td
    u  = up * mf

    P = np.zeros(shape=(n,n,Nz,n))
    # debug:
    RSD1 = np.zeros(shape=(n,n,Nz,n))
    RSD2 = np.zeros(shape=(n,n,Nz,n))

    RD1 = np.zeros(shape=(n,n,Nz,n))
    RS1 = np.zeros(shape=(n,n,Nz,n))

    RD2 = np.zeros(shape=(n,n,Nz,n))
    RS2 = np.zeros(shape=(n,n,Nz,n))

    Phi1 = np.zeros(shape=(n,n,Nz,n))
    Phi2 = np.zeros(shape=(n,n,Nz,n))
    Phi0 = np.zeros(shape=(n,n,Nz,n))

    for vp in range(0,n):
        v = vp * mf
        for rxp in range(0,n):
            rx = rxp*mf
            for ryp in range(0,n):
                ry=ryp*mf
                for rzp in range(0,Nz):
                    rz=(rzp+0.5)*mf
                    # rz=rzp *mf
                    rs1 = np.sqrt((u-rx)**2+(v-ry)**2+(0-rz)**2)
                    rs2 = np.sqrt((u-rx)**2+(v-ry)**2+(0-z0-2*zb-rz)**2)
                    rd1 = np.sqrt((rx-s)**2+(ry-t)**2+(0+z0-rz)**2)
                    rd2 = np.sqrt((rx-s)**2+(ry-t)**2+(0-z0-2*zb-rz)**2)
                    rsd1 = np.sqrt((td*mf)**2+(v-t)**2+(0+z0-0)**2)
                    rsd2 = np.sqrt((td*mf)**2+(v-t)**2+(0-z0-2*zb-0)**2)

                    phi_2 = math.exp(-beta*rd1)/rd1-math.exp(-beta*rd2)/rd2
                    phi_1 = (math.exp(-beta*rs1)/rs1-math.exp(-beta*rs2)/rs2)
                    phi_0 = (math.exp(-beta*rsd1)/rsd1-math.exp(-beta*rsd2)/rsd2)

                    #debug:
                    RSD1[rxp,ryp,rzp,vp]= rsd1
                    RSD2[rxp,ryp,rzp,vp]= rsd2
                    RD1[rxp,ryp,rzp,vp] = rd1
                    RD2[rxp,ryp,rzp,vp] = rd2
                    RS1[rxp,ryp,rzp,vp] = rs1
                    RS2[rxp,ryp,rzp,vp] = rs2

                    Phi0[rxp,ryp,rzp,vp]= phi_0
                    Phi1[rxp,ryp,rzp,vp]= phi_1
                    Phi2[rxp,ryp,rzp,vp]= phi_2

                    num = phi_1*phi_2
                    P[rxp,ryp,rzp,vp] = num/phi_0

    Gs = np.average(P,axis=3)/(4*math.pi*D)
    return Gs*(mf)**3, RSD1, RSD2, RD1, RD2, RS1, RS2, Phi0, Phi1, Phi2
    # return Gs, RSD1, RSD2, RD1, RD2, RS1, RS2, Phi0, Phi1, Phi2

def getKernel_th(td, mf, n,  Nz, scatter_prop):
    '''
    Get kernel for the dipole model
    DONE: #check with getKernel()
    scatter_prop = {'g', 'mua', 'musp', 'D', 'Reff', 'beta'}
    '''

    g, mua, musp, D, Reff, beta = \
            scatter_prop['g'], scatter_prop['mua'], \
            scatter_prop['musp'], scatter_prop['D'], \
            scatter_prop['Reff'], scatter_prop['beta'] 

    P = torch.zeros(n, n, Nz, n) # dimension: rx, ry, rz, v
    z0 = 1/(musp+mua)
    zb = (1+Reff)/(1-Reff)*2*D
    Gs = torch.zeros( n, n, Nz)
    s = n/2*mf         # detector col
    t = n/2*mf         # detector row
    u = (n/2-td) * mf  # source col 

    vec_v, vec_rx, vec_ry, vec_rz = torch.arange(0, n, dtype = torch.float64) * mf, \
                                    torch.arange(0, n, dtype = torch.float64) * mf, \
                                    torch.arange(0, n, dtype = torch.float64) * mf, \
                                    torch.arange(0, Nz,dtype = torch.float64) * mf

    vec_rz += 0.5 * mf 
    vec_v.unsqueeze_(0).unsqueeze_(0).unsqueeze_(0)
    vec_rx.unsqueeze_(1).unsqueeze_(2).unsqueeze_(3)
    vec_ry.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
    vec_rz.unsqueeze_(0).unsqueeze_(0).unsqueeze_(3)

    V, Rx, Ry, Rz = vec_v.expand_as(P), vec_rx.expand_as(P), vec_ry.expand_as(P), vec_rz.expand_as(P) 
    RS1 = torch.sqrt( (u - Rx)**2 + (V-Ry)**2 + (0+z0-Rz)**2)
    # RS1 = torch.sqrt( (u - Rx)**2 + (V-Ry)**2 + (0-Rz)**2)
    RS2 = torch.sqrt( (u - Rx)**2 + (V-Ry)**2 + (0-z0-2*zb-Rz)**2)
    RD1 = torch.sqrt( (Rx-s)**2 + (Ry-t)**2 + (0+ z0 - Rz)**2 )
    # RD1 = torch.sqrt( (Rx-s)**2 + (Ry-t)**2 + (0+Rz)**2 )
    RD2 = torch.sqrt( (Rx-s)**2 + (Ry-t)**2 + (0-z0-2*zb-Rz)**2 ) 

    RSD1 = torch.sqrt( (u-s)**2 + (V-t)**2 + (0+z0-0)**2 )
    RSD2 = torch.sqrt( (u-s)**2 + (V-t)**2 + (0-z0-2*zb-0)**2 )

    PHI_0 = torch.exp( -beta*RSD1 ) / RSD1 - torch.exp(-beta*RSD2) / RSD2
    PHI_1 = torch.exp( -beta*RS1 ) / RS1 - torch.exp( -beta*RS2) / RS2
    PHI_2 = torch.exp( -beta*RD1 ) / RD1 - torch.exp( -beta*RD2) / RD2
    P = PHI_1 * PHI_2 / PHI_0 

    # Use slit light source #
    Gs = torch.mean( P, dim = 3 ) / ( 4 * math.pi * D )

    # normalize Gs #
    # Gs = Gs / Gs.sum() 
    return Gs*(mf**3), RSD1, RSD2, RD1, RD2, RS1, RS2,  PHI_0, PHI_1, PHI_2
    
def getIntensity(td):

    K0=[]    
    t = Nc/2*mf 
    # TODO : the unit in z , does it match with the units in the other two directions ? 
    for vp in range(0,Nc):
        v = vp * mf
        z0 = 1/(musp+mua);
        zb = (1+Reff)/(1-Reff)*2*D;
        rsd1 = np.sqrt((td*mf)**2+(v-t)**2+(0+z0-0.5 * mf)**2)
        rsd2 = np.sqrt((td*mf)**2+(v-t)**2+(0-z0-2*zb-0.5 * mf )**2)
        phi_0 = 1/(4*math.pi*D)*(math.exp(-beta*rsd1)/rsd1-math.exp(-beta*rsd2)/rsd2)
        K0.append(phi_0)
    I0 = (np.average(np.array(K0)))

    return I0

   
def main():
    src_row = 20
    n = Nz
    Id=[]
    Ia = []

    scatter_prop = { 'g': g, 'mua':mua, 'musp':musp, 'D': D, 'Reff':Reff, 'beta': beta  } 
    assert is_slit == 1

    for d in range(21, 81):
        det_row = d
        det_col = Nc/2
        td = det_row - src_row
        I0 = getIntensity(td)

        print('td =%d pixels'%(td ) )

        # Gs, RSD1, RSD2, RD1, RD2, RS1, RS2, Phi0, Phi1, Phi2 = getKernel(td,n)
        Gs_th, RSD1_th, RSD2_th, RD1_th, RD2_th, RS1_th, RS2_th, Phi0_th, Phi1_th, Phi2_th = \
                getKernel_th( td, mf, n, Nz, scatter_prop )

        # Gs_th = Gs_th.cpu().numpy()
        # RSD1_th = RSD1_th.cpu().numpy()
        # RSD2_th = RSD2_th.cpu().numpy()
        # RD1_th = RD1_th.cpu().numpy()
        # RD2_th = RD2_th.cpu().numpy()
        # RS1_th = RS1_th.cpu().numpy()
        # RS2_th = RS2_th.cpu().numpy()
        # Phi1_th = Phi1_th.cpu().numpy()
        # Phi2_th = Phi2_th.cpu().numpy()
        # Phi0_th = Phi0_th.cpu().numpy() 
        # print('diff Gs: %f'%(np.abs(Gs_th - Gs).max() ) )
        # print('diff RSD1: %f'%( np.abs(RSD1_th- RSD1).max() ))
        # print('diff RSD2: %f'%(np.abs(RSD2_th- RSD2).max())  )
        # print('diff RD1: %f'%(np.abs(RD1_th- RD1).max() ))
        # print('diff RD2: %f'%(np.abs(RD2_th- RD2).max() ))
        # print('diff RS1: %f'%(np.abs(RS1_th- RS1).max() ))
        # print('diff RS2: %f'%(np.abs(RS2_th- RS2).max() ))
        # print('diff Phi0: %f'%(np.abs(Phi0_th- Phi0).max() ))
        # print('diff Phi1: %f'%(np.abs(Phi1_th- Phi1).max() ))
        # print('diff Phi2: %f'%(np.abs(Phi2_th- Phi2).max() ))
        # hstack_cmp = np.hstack( (Gs[:,:, :], Gs_th[:, :, :]) )

        V_element = Vcon[int(det_row-n/2):int(det_row+n/2),int(det_col-n/2):int(det_col+n/2),:]
        absorb = np.multiply(Gs_th,V_element)

        # Raytov 
        intensity = I0*np.exp(-absorb.sum())
        print( absorb.sum() )
        # Bonn 

        Id.append(math.log10(intensity))

    scipy.io.savemat('intensity_dipole_slit.mat',mdict={'Id': Id}) 
          
if __name__ == '__main__':
    main()
