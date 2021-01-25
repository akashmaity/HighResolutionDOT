# multiplex (demultiplex) module # 
import numpy as np
import scipy.io as sio
import scipy.interpolate as sinterp
import scipy.misc as smisc
import PIL.Image as image

# pytorch module
import torch
import torch.optim as optim 
import torchvision.transforms.functional as tv_func
import torch.nn.functional as torch_func
from torch.autograd import Variable

import pyProcImg.procImg as procImg 
import matplotlib.pyplot as plt 


def DeMultiplex_gl_1d( Imgs, T_d, laser_prof_info, kernel_size, 
        Ts_init=None, gl_rmax=20, max_it = 10000, lr = .001 ,
        gl_debug = None):
    '''
    de-multiplex the 1D gl profile (the laser width):

    Img = conv2d( Ts, g_l ), 
    g_l is the laser profile
    Ts is the light tranport images (row-wise, not point wise)

    Given img, g_l, estimate Ts

    inputs:
        Imgs - Nimg x H x W tensor,  
               captured images. NOTE: all elements in IMGS should be float

        T_d -  a.k.a delta_v, the delay between the exposure line and projection line
               1D array of t_d corresponding to Imgs, length should equal Nimg 

        laser_prof_info - 
            {   'laser_profile': 1D laser profile, the length of the 
            laser profile determines the # channels in T
                't_values': t_values corresponds to laser_profile  } 

        kernel_size - the length of the 1D g_l profile

        Ts_init - C x H x W, where C = len(laser_prof_info['laser_profile'])

        gl_rmax - the radius of the gl, values in g_l outside the radius would be 0

        max_it - iteration # for optimization

    outputs:
        Ts - T(t, v)
    ''' 
    nImgs, H, W = Imgs.shape[0], Imgs.shape[1], Imgs.shape[2]
    laser_prof = laser_prof_info['laser_profile'].flatten()
    t_values = laser_prof_info['t_values'].flatten() 

    # Asserts #
    C_out = nImgs
    C = kernel_size
    assert len(T_d) == C_out
    assert len(laser_prof) == len(t_values) == C

    # Initilize Ts #
    device = torch.device('cuda')
    if Ts_init is not None:
        assert Ts_init.shape[0] == C and Ts_init.shape[1] == H and Ts_init.shape[2] == W
        Ts_init.unsqueeze_(0)
        Ts = Variable(Ts_init, requires_grad=True)
    else:
        Ts = torch.randn(1, C, H, W, requires_grad = True, device="cuda")

    # Prepare for the kernels and Imgs #

    if gl_debug is not None:
        g_l = gl_debug
    else:
        g_l = torch.zeros(C_out, C, 1, 1).cuda()
        laser_prof = laser_prof.astype(np.float)
        for id_td, t_d in enumerate(T_d): 
            g_l_per = procImg.get_laser_profile_td( 
                    laser_prof, t_values.astype(np.float), 
                    t_d=t_d, radius=gl_rmax, kernel_size=kernel_size, 
                    fill_value = (laser_prof[0], laser_prof[-1] ))
            g_l[id_td, :, 0, 0] = torch.from_numpy(g_l_per).cuda() 

#    import ipdb
#    ipdb.set_trace()

    # optimization # 
    optimizer = optim.Adam( [Ts], lr= lr, betas=(.9, .999) )
    Imgs = Imgs.float() 
    check_step= 500
    for it in range(max_it):
        optimizer.zero_grad() 
        Imgs_syn = torch_func.conv2d(Ts, g_l).squeeze() 
        diff = Imgs_syn - Imgs
        loss = torch.sum(diff**2 ) 
        loss.backward() 
        # print loss function #
        if it % check_step == 0:
            print('iter %d/%d, loss = %.5f'%( it, max_it , loss.data.cpu().numpy() ) ) 
            # termination #
            if it > check_step:
                loss_diff = np.abs(loss_prev - loss.data.cpu().numpy())
                if loss_diff <= 0.1:
                    print('error change is small, converged !'); break 
            loss_prev = loss.data.cpu().numpy() 
        optimizer.step() 

    print('iter %d/%d, loss = %.5f'%( it+1, max_it , loss.data.cpu().numpy() ) ) 
    return Ts.detach().cpu(), g_l.cpu()
