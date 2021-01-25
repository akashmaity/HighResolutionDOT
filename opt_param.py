import torch
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import scipy.io as sio
import scipy.io as sio 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import *

def getIntensity_th_v2( td, n, mf, props, if_avg = True ):
    ''' 
    Get I0 using diffusion approximation and dipole solution
    Inputs:
        td - an array of delays
        n - # of cols of the projected line (== # of cols of images for simulation)
        props - [D, beta] : the optical properties for the homogenous medium
        mf -  the metric size (mm) of one pixel
            the magnification factors in different dimensions [mf_uv, mf_st, mf_z ], 
             magnification factors in the uv, st (camera coordinate) dimensions
    ''' 
    Reff = 0.496
    z0 = 3 * props[0]
    zb = (1+Reff)/(1-Reff)*2* props[0];
    zv = z0+4*props[0]
    t = n/2 * mf[1] 
    V = torch.arange(0, n).type_as(props) * mf[0] 
    V = V.to( props.device ).unsqueeze(0)

    # TODO: is 0.5 here needed in order to make the optimization of homogeneous parameter to work ?
    RSD1 = torch.sqrt(( td.unsqueeze(1) * mf[1] )**2 + (V - t)**2 + (3 * props[0] - 0.5)**2 )  
    RSD2 = torch.sqrt(( td.unsqueeze(1) * mf[1] )**2 + (V - t)**2 + (-3 * props[0] -2* 2*props[0]-0.5 )**2)

    #Phi0 = 1/( 4*math.pi) \
            #*( z0*(props[1]*RSD1+1)*torch.exp(-props[1]*RSD1)/(RSD1**3) + zv*(props[1]*RSD2+1)*torch.exp(-props[1]*RSD2)/(RSD2**3))
    Phi0 = props[2]/( 4*math.pi * props[0] ) \
            * ( torch.exp( -props[1] * RSD1) / RSD1 - torch.exp( -props[1]*RSD2 ) / RSD2 )     
    if if_avg:
        I0 = torch.mean( Phi0, dim=1 )
    else:
        I0 = torch.sum( Phi0, dim=1 )
    
   
    return I0


def opt_scatter_prop_homo(R_m, props, Tds, n_col, mf, step, max_iter=500, Td_mask = None):
    '''
    R_m - 1D vector of measurements for homogeneous region 
    '''

    props_opt = props.cuda().requires_grad_()
    optimizer = optim.Adam([props_opt], lr = step, betas= (.9, .99)) 
    Tds_gpu = Tds.type_as(R_m).cuda() 

    print() 
    for it in range( max_iter ):
        optimizer.zero_grad() 
        R_d = getIntensity_th_v2(Tds_gpu, n_col, mf, props_opt , if_avg = True) 

        diff_imgs = R_d - R_m 

        if Td_mask is not None:
            diff_imgs = diff_imgs * Td_mask 

        loss = (diff_imgs**2).sum().sqrt()
        loss.backward()
        optimizer.step() 
        print( 'iter: %d/%d, loss: %.5f'%(it, max_iter, loss.data.cpu().numpy() ), end='\r' ) 

    print()

    return props_opt
 
def main():
    max_iter = 1
    max_iter_props = 5000
    max_iter_Q = 300
    step_props = .001 
    step_Q = .001 # .0001
    
    N = 100
    Nz = 33


    Nr, Nc = N, N
    Nxy = N 

    mf = torch.FloatTensor([1, 1, 1])

    fname = '/home/akm8/Diffuse_Model/diff_reflectance.mat' 
    mat = sio.loadmat(fname) 
    R_m = mat['Intensity'].flatten()
    Tds = torch.FloatTensor(mat['dTs']).flatten()
    R_m = torch.FloatTensor(R_m).cuda()
    
    props_init = torch.FloatTensor([0.34192, 0.1571, 0.08792]).cuda()
    
    td_min, td_max  = 5, 45
    Td_mask = torch.zeros(len(Tds))
    Td_mask[td_min:td_max] = 1  
    Td_mask = Td_mask.cuda()
    
    # opt loop #
    for it in range(0, max_iter):
        # optimize for scatter props #
        print('\n')
        print('---------')
        print('iteration: %d of %d: \n'%( it+1,  max_iter ))

        # optimize for scatter properties # 
        print('\nopt. scattering props ...')
        props_init_ = props_init.clone() 
        
        opt_paras = opt_scatter_prop_homo(\
                            R_m, props_init, Tds, Nxy,mf, 
                            step= step_props * (.9)**it,
                            max_iter = max_iter_props, Td_mask = Td_mask).detach() 

        print( "init props: %.5f, %.5f, %.5f"%( props_init_[0], props_init_[1], props_init_[2] ) )
        print( "opt  props: %.5f, %.5f, %.5f"%( opt_paras[0], opt_paras[1], opt_paras[2]) )
        print('\nopt. scattering props done')

    print('** \n')

if __name__ == '__main__':
    main()