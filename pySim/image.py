# Module to simulate images #
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import scipy.io as sio
import scipy.io as sio 
import matplotlib.pyplot as plt 

from matplotlib.pyplot import *

def _norm2(x, y, z):
    return torch.sqrt( x**2 + y**2 + z**2 )

def getKernel_th( td, mf, n, Nz, scatter_prop):
    '''
    Get kernel for the dipole model
    DONE: #check with getKernel()

    inputs:
    mf - the magnification factors in different dimensions [mf_uv, mf_st, mf_z ], magnification factors in the uv, st (camera coordinate) dimensions

    scatter_prop = {'g', 'mua', 'musp', 'D', 'Reff', 'beta'}
    '''
    
    if not isinstance( mf, list):
        mf = [mf, mf, mf]

    mf_uv, mf_st, mf_z = mf[0], mf[1], mf[2]
    g, mua, musp, D, Reff, beta = scatter_prop['g'], scatter_prop['mua'], scatter_prop['musp'], scatter_prop['D'], scatter_prop['Reff'], scatter_prop['beta'] 
    P = torch.zeros(n, n, Nz, n)# dimension: rx, ry, rz, v
    z0 = 1/(musp+mua)
    zb = (1+Reff)/(1-Reff)*2*D
    
    zv = z0+2*zb
    Gs = torch.zeros( n, n, Nz)
    s = (n-1)/2* mf_st              # detector col
    t = (n-1)/2* mf_st              # detector row
    u = ((n-1)/2-td) * mf_uv        # source col 

    vec_v, vec_rx, vec_ry, vec_rz = torch.arange(0, n, dtype = torch.float32) * mf_uv, \
                                    torch.arange(0, n, dtype = torch.float32) * mf_st, \
                                    torch.arange(0, n, dtype = torch.float32) * mf_st, \
                                    torch.arange(0, Nz,dtype = torch.float32) * mf_z

    vec_rz += 0.5 * mf_z

    vec_v.unsqueeze_(0).unsqueeze_(0).unsqueeze_(0)
    vec_rx.unsqueeze_(1).unsqueeze_(2).unsqueeze_(3)
    vec_ry.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
    vec_rz.unsqueeze_(0).unsqueeze_(0).unsqueeze_(3)

    V, Rx, Ry, Rz = vec_v.expand_as(P), vec_rx.expand_as(P), vec_ry.expand_as(P), vec_rz.expand_as(P) 
    # RS1 = torch.sqrt( (u - Rx)**2 + (V-Ry)**2 + (0+z0-Rz)**2)
    RS1 = torch.sqrt( (u - Rx)**2 + (V-Ry)**2 + (0+z0-Rz)**2)
    RS2 = torch.sqrt( (u - Rx)**2 + (V-Ry)**2 + (0-z0-2*zb-Rz)**2)
    RD1 = torch.sqrt( (Rx-s)**2 + (Ry-t)**2 + (Rz - 0)**2 )
    RD2 = torch.sqrt( (Rx-s)**2 + (Ry-t)**2 + (0-2*zb-Rz - 0)**2 ) 

    RSD1 = torch.sqrt( (u-s)**2 + (V-t)**2 + (0+z0-0)**2 )
    RSD2 = torch.sqrt( (u-s)**2 + (V-t)**2 + (0-z0-2*zb-0)**2 )
    
    PHI_0 = 1/ ( 4 * math.pi)*(z0*(beta*RSD1+1)*torch.exp(-beta*RSD1)/(RSD1**3)+zv*(beta*RSD2+1)*torch.exp(-beta*RSD2)/(RSD2**3))
    PHI_1 = 1/ ( 4 * math.pi * D)* (torch.exp( -beta*RS1 ) / RS1 - torch.exp( -beta*RS2) / RS2)
    PHI_2 = 1/ ( 4 * math.pi)*(Rz*(beta*RD1+1)*torch.exp(-beta*RD1)/(RD1**3)+(Rz+2*zb)*(beta*RD2+1)*torch.exp(-beta*RD2)/(RD2**3))

    P = PHI_1 * PHI_2  / PHI_0

    # Use slit light source #
    Gs = torch.mean( P, dim = 3 )

    # normalize Gs #
    # Gs = Gs / Gs.sum() 
    return Gs*( mf_st**2 * mf_z )
    
def getIntensity(td, n, mf, scatter_prop):
    ''' Get I0 for the diple model 
    scatter_prop = {'g', 'mua', 'musp', 'D', 'Reff', 'beta'}

    mf - the magnification factors in different dimensions [mf_uv, mf_st, mf_z ], magnification factors in the uv, st (camera coordinate) dimensions

    ''' 
    g, mua, musp, D, Reff, beta = scatter_prop['g'], scatter_prop['mua'], scatter_prop['musp'], scatter_prop['D'], scatter_prop['Reff'], scatter_prop['beta']
    z0 = 1/(musp+mua); # 3 * D 
    zb = (1+Reff)/(1-Reff)*2*D; # 2 * D

    K0 = []
    t = n/2 * mf[1]
    for vp in range(0,n):
        v = vp * mf[0]
        # rsd1 = np.sqrt((td * mf[1] )**2+(v-t)**2+(0+z0-0.5 * mf[2])**2)
        # rsd2 = np.sqrt((td * mf[1] )**2+(v-t)**2+(0-z0-2*zb-0.5 * mf[2] )**2)
        rsd1 = np.sqrt((td * mf[1] )**2+(v-t)**2+(0+z0-0.5 )**2)
        rsd2 = np.sqrt((td * mf[1] )**2+(v-t)**2+(0-z0-2*zb-0.5  )**2)
        phi_0 = 1/(4*math.pi*D)*(math.exp(-beta*rsd1)/rsd1-math.exp(-beta*rsd2)/rsd2)
        K0.append(phi_0) 
    I0 = np.mean(np.array(K0)) 

    return I0 

def getIntensity_th( td, n, mf, props ):
    ''' Get I0 using diffusion approximation and dipole solution
    Inputs:
        props - [D, beta] : the optical properties for the homogenous medium
        mf - the magnification factors in different dimensions [mf_uv, mf_st, mf_z ], 
             magnification factors in the uv, st (camera coordinate) dimensions
    ''' 
    # z0, zb = 3 * props[0], 2 * props[0]
    z0 = 3 * props[0]
    zb = (1+props[2])/(1-props[2])*2*props[0]
    zv = z0+2*zb
    t = n/2 * mf[1] 
    V = torch.arange(0, n).type_as(props) * mf[0] 
    V = V.to( props.device ) 
    RSD1 = torch.sqrt(( td * mf[1] )**2 + (V - t)**2 + (3 * props[0] - 0.5  )**2 )  
    RSD2 = torch.sqrt(( td * mf[1] )**2 + (V - t)**2 + (- 3 * props[0] -2* zb-0.5 )**2 )
    Phi0 = 1/(4*math.pi)*(z0*(props[1]*RSD1+1)*torch.exp(-props[1]*RSD1)/(RSD1**3)+zv*(props[1]*RSD2+1)*torch.exp(-props[1]*RSD2)/(RSD2**3)) 
    I0 = Phi0.mean() 

    # TODO: shall we rescale the output to account for mf ? # 

    return I0


def dmap2Qvolume( dmap, D ):
    ''' Given depth map (H x W) , get a occupancy volume (D x H x W) 
    Inputs:
        dmap -  H x W depth map. The depth value should be integers and
                dmap.max() <= D. For depth=0, this means that there is no occupacy at
                any depth at (x,y)
        D - The depth dimension of Q. the occupancy volume Q has size [D, H, W]
    ''' 
    assert dmap.dtype is torch.int
    assert D >= dmap.max() and dmap.min() >= 0 

    H, W = dmap.shape[0], dmap.shape[1]
    Q = torch.zeros( D, H, W ) 

    nonzero_loc = torch.nonzero( dmap )
    for loc in nonzero_loc:
        irow, icol = loc[0], loc[1]
        d = dmap[ irow, icol ]
        Q[d, irow, icol] = 1

    return Q

def conv3d(Q, gs, padding = 0):
    '''convolve Q with gs
    Inputs:
        Q - [D, H, W]
        gs - [D, width, width]
    ''' 
    return F.conv2d( Q.unsqueeze(0), gs.unsqueeze(0), padding = padding )

def conv3d_multiple_gs(Q, Gs, padding = 0):
    return F.conv2d( Q.unsqueeze(0), Gs, padding = padding ) 

def loss_norm2(Img, gs, Q):
    ''' norm2 loss
    Imgs -  H x W
    gs   -  D x w x w (w: kernel widith)
    Q    -  D x H x W '''

    assert gs.shape[1] == gs.shape[2]
    padding_R = int( gs.shape[1]/2 )
    diff_img = conv3d(Q, gs, padding = padding_R) - Img
    return torch.sum(diff_img**2)

def loss_non_neg(Q_var):
    loss = torch.norm( Q_var[Q_var<0], p=1 )
    return loss 

def loss_norm2_multi(Imgs, Gs, I0, Q, mask=True, approx = 'Bonn', use_diffimg = False, Td_mask = None, maskI = None):
    ''' norm2 loss
    Imgs -  n_delay x H x W
    Gs   -  n_delay x D x w x w (w: kernel widith)
    Q    -  D x H x W
    I0   -  n_delay x H x W

    OPTIONAL
    mask - if maskout the zero values in measurement
    approx - name for the approximation: {'Bonn', 'Raytov'} 
    use_diffimg - if use difference image for modeling. If true, then Gs is the difference kernel, Imgs is the corresponding diff images
    Td_mask - a vector of 0 and 1, corresponding if mask out Td values
    '''

    padding_R = int( Gs.shape[2]/2 ) 

    # import ipdb
    # ipdb.set_trace()

    # Forward model #

    if approx == 'Bonn':
        # Bonn #
        if not use_diffimg:
            diff_imgs =(I0 - I0 * conv3d_multiple_gs(Q, Gs, padding = padding_R).squeeze()) - Imgs 
        else: 
            diff_imgs = I0 * conv3d_multiple_gs(Q, Gs, padding = padding_R).squeeze() - Imgs

    else: # Raytov # 
        if not use_diffimg:
            imgs_sim = I0 * torch.exp( - conv3d_multiple_gs(Q, Gs, padding = padding_R).squeeze() )
            diff_imgs = imgs_sim - Imgs
        else:
            diff_imgs = conv3d_multiple_gs( Q, Gs, padding = padding_R ).squeeze() - Imgs

    if mask:
        if maskI is None:
            diff_imgs[ Imgs==0 ] = 0 
        elif maskI is not None :
            diff_imgs[ maskI==1 ] = 0

    if Td_mask is not None:
        # mask out some tds
        assert(len(Td_mask) == Imgs.shape[0]) 
        diff_imgs = diff_imgs * Td_mask.unsqueeze(1).unsqueeze(2)

    # Loss function (norm2) # 
    return torch.sum( diff_imgs **2 ).sqrt()

def opt_dmap( Imgs, Gs, step, max_iter = 100, dmap_init = None, z_min=1, sigma = .1, lambdas = [0.] ):
    '''optimize dmap
    Imgs -  n_delay x H x W
    Gs   -  n_delay x D x w x w (w: kernel widith)
    '''
    D, H, W = Gs.shape[1], Imgs.shape[1], Imgs.shape[2]
    if dmap_init is None:
        dmap_init = torch.randn( (1, H, W), device = Imgs.device) + 2
        dmap_var = dmap_init.requires_grad_()
    # dmap_var = dmap_init.requires_grad_()

    vz = torch.linspace( z_min , D, D, device = dmap_var.device)
    Q_z = vz.unsqueeze(1).unsqueeze(1).repeat([1, H, W]) # Q_z size = [D, H, W]
    
    # optimizer = optim.LBFGS([dmap_var], lr = step)
    optimizer = optim.Adam([dmap_var], lr = step, betas=[ .9, .99 ] )
    for it in range( max_iter ):
        # LBFGS 
        # loss =0.
        # def closure():
        #     optimizer.zero_grad() 
        #     imgs_simulate = depth_to_imgs( dmap_var, Gs, z_min = z_min, sigma = sigma, Q_z = Q_z )
        #     loss = torch.sum( (imgs_simulate - Imgs.unsqueeze(0) )**2 )
        #     # import ipdb
        #     # ipdb.set_trace()
        #     loss.backward() 
        #     print( 'iter: %d/%d, loss: %f'%( it, max_iter, loss.data.cpu().numpy() ) )
        #     return loss 
        # optimizer.step( closure )

        # Adam
        optimizer.zero_grad()
        imgs_simulate = depth_to_imgs( dmap_var, Gs, z_min = z_min, sigma = sigma, Q_z = Q_z )
        loss = torch.norm( imgs_simulate - Imgs.unsqueeze(0), p=2) + lambdas[0] * torch.norm( dmap_var, p=0 )
        loss.backward()
        optimizer.step()
        if it % 50 ==0:
            print( 'iter: %d/%d, loss: %f'%( it, max_iter, loss.data.cpu().numpy() ) )


    return dmap_var.data, imgs_simulate.detach().cpu()

def opt_scatter_prop( I0_m, props, Q, Tds, n, mf,
                      step, Td_mask = None, max_iter = 500):
    ''' optimize for scattering properties 
    props - initial value for props, list: [D, beta]
    '''

    props_opt = torch.FloatTensor(props, device = 'cuda', requires_grad = True)
    I0_r = torch.zeros( len(Imgs), device = 'cuda' )
    
    optimizer = optim.Adam([props_opt], lr = step, betas= (.9, .99) )

    for it in range( max_iter ):
        optimizer.zero_grad()

        # TODO: parallelize this
        for i in range( len(Tds) ):
            td = Tds[i]
            I0_r[i] = getIntensity_th(td, n, mf, props) 

        loss = ((I0_r - I0_m)**2 ).sum().sqrt() 
        loss.backward()
        optimizer.step() 

        if it % 10 == 0:
            print( 'iter: %d/%d, loss: %f'%(it, max_iter, loss.data.cpu().numpy() ) , end = '\r') 

    return props_opt


def opt_Q( Imgs, Gs, I0, step, Td_mask = None, max_iter = 100, Q_init = None, lambdas = [0.] , 
            opt_name = 'LBFGS', model = 'Bonn', mask = None, use_diffimg = False):
    '''optimize for Q
    Imgs -  n_delay x H x W
    Gs   -  n_delay x D x w x w (w: kernel widith)
    I0   -  n_delay x H x W, the background image if there is no in-homonegeous tissues inside 
    step -  step size

    Optional inputs:
    max_iter - maximal iterations
    Q_init - initial Q, zeros if not set
    lambdas - [lambdas] for [L1_norm(Q), non_negative(Q)]
    ''' 

    # opt_name = 'LBFGS'#'Adam' # 'LBFGS'
    
    D, H, W = Gs.shape[1], Imgs.shape[1], Imgs.shape[2]

    if Q_init is None:
        Q_var = torch.zeros( (D, H, W), requires_grad = True , device='cuda')
    else:
        Q_var = Q_init.requires_grad_() 

    print('Q size: [D, H, W] = %d, %d, %d'%(D, H, W ) )
    print('Gs size: [nDelay, D, w, w] = %d, %d, %d, %d'%(len(Imgs), D, Gs.shape[2], Gs.shape[3]) )
    print('Imgs size: [nDelay, D, H, W] = %d, %d, %d, %d'%( len(Imgs), D, H, W ) ) 

    if opt_name == 'Adam':
        optimizer = optim.Adam([Q_var], lr = step, betas= (.9, .99) )
    elif opt_name == 'LBFGS':
        optimizer = optim.LBFGS([Q_var], lr = step)

    for it in range( max_iter ): 
        if opt_name == 'Adam':
            # Adam optimizer #
            optimizer.zero_grad() 

            loss = loss_norm2_multi(Imgs, Gs, I0,  Q_var, mask = True, Td_mask = Td_mask, approx = model, maskI = mask, use_diffimg = use_diffimg) \
                    + lambdas[0] * torch.norm( Q_var.reshape([1, -1]).squeeze(), p= 1 ) + lambdas[1] * loss_non_neg(Q_var)

            # loss =  lambdas[0] * torch.norm( Q_var.reshape([1, -1]).squeeze(), p=0 ) \
            #         + lambdas[1] * loss_non_neg(Q_var)

            loss.backward()
            optimizer.step() 
            if it % 50 == 0:
                # print( 'iter: %d/%d, loss: %f'%(it, max_iter, loss.data.cpu().numpy() ) , end = '\r') 
                print( 'iter: %d/%d, loss: %0.5f'%(it, max_iter, loss.data.cpu().numpy() ) , ) 

        elif opt_name == 'LBFGS':
            # LBFGS optimization #
            def closure():
                optimizer.zero_grad() 
                loss = loss_norm2_multi(Imgs, Gs, I0, Q_var, mask=True, Td_mask = Td_mask, approx = model, maskI = mask, use_diffimg = use_diffimg) \
                        + lambdas[0] * torch.norm(Q_var, p=1)  \
                        + lambdas[1] * loss_non_neg( Q_var ) 

                loss.backward() 
                print( 'iter: %d/%d, loss: %f'%(it, max_iter, loss.data.cpu().numpy() ), end = '\r' ) 
                return loss 

            optimizer.step( closure ) 

    return Q_var.data

def adjust_lr(optimizer, step):
    for para_group in optimizer.param_groups:
        para_group['lr'] = step
    return optimizer

def Q_to_imgs(Q, Gs, I0, z_min = 1, approx = 'Bonn'):
    '''
    Inputs:
        Gs   -  n_delay x D x w x w (w: kernel widith)
        Q    -  D x H x W
    '''
    padding_R = int( Gs.shape[2]/2 ) 
    if approx == 'Bonn': #Bonn #
        imgs = I0 * ( 1 - conv3d_multiple_gs(Q, Gs, padding = padding_R).squeeze() )
    else: # Raytov 
        imgs = I0 * torch.exp( - conv3d_multiple_gs(Q, Gs, padding = padding_R).squeeze() ) 
    return imgs
    

def depth_to_imgs(dmap, Gs, z_min=1, sigma=1, Q_z = None):
    ''' Given depth map and kernel gs, get the images
    Inputs:
        Gs   -  n_delay x D x w x w (w: kernel widith)
        dmap -  1 x H x W
    Outputs:
        imgs -  n_delay x H x W
    '''
    D, H, W, w, n_delay = Gs.shape[1], dmap.shape[0], dmap.shape[1], Gs.shape[2], Gs.shape[0] 
    padding_R = int( w/2 ) 

    # dmap to Q #
    if Q_z is None:
        vz = torch.linspace( z_min , D, D, device = dmap.device)
        Q_z = vz.unsqueeze(1).unsqueeze(1).repeat([1, H, W]) # Q_z size = [D, H, W]
        Q = torch.exp( -( Q_z - dmap )**2 / (sigma**2)  )
        # Q to imgs #
        imgs = conv3d_multiple_gs(Q, Gs, padding = padding_R)

    else:
        # Q to imgs #
        Q = torch.exp( -( Q_z - dmap )**2 / (sigma**2)  )
        # Q = torch.zeros( (D, H, W), device = Q_z.device )
        # for i in range(D):
        #     Q[i , :, :] = torch.exp( -(Q_z[i, :, :] - dmap)**2 /(sigma**2)  )
        imgs = conv3d_multiple_gs(Q, Gs, padding = padding_R) 

    return imgs
