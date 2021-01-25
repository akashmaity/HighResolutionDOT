

# inspect the spatial temporal gradients #
import torch
import numpy as np
import math
import scipy.io as sio
import pySim.image as pySim
from matplotlib.pyplot import * 

# global variables #
Nr, Nc, Nz = 100, 100, 33
g = 0.9 
mua = 0.045
musp = 35.65 * (1-g) #35.65*(1-g)
D = 1/(3*(mua+musp)) 
Reff = 0 
beta = math.sqrt(3*musp*mua) 

# Multiplication factor 1 px = mf mm
mf = 0.1 

def main():
    # set parameters #
    scatter_prop = {'g':g, 'mua':mua, 'musp':musp, 'D':D, 'Reff':Reff, 'beta':beta}
    TD = 121 

    array_grad_contrast = []
    D_VEIN = np.arange( 4, 30) 
    for d in D_VEIN:
        d_vein = d
        mat = sio.loadmat('./dat/dipole_validate/img_dt_dvein_6_rvein_1.mat')
        Imgs, Td = mat['IMGS_td'], mat['dTs'].flatten() 

        # Optimizing || I_m - I0 (1 - Q \conv gs) ||2 + other constriants #
        Imgs = torch.FloatTensor( Imgs ).permute(2, 0, 1)
        n_delay, H, W = Imgs.shape[0], Imgs.shape[1], Imgs.shape[2] 

        # get Gs: n_delay x D x w x w #
        # get I0: n_delay x H x W     #
        I0, Gs = [], []
        for td in Td:
            print( 'getting I0 and Gs for td=%f (Td_max= %f)'%( td, Td.max() ), end='\r' )
            I0.append( pySim.getIntensity( -td, Nz, mf, scatter_prop) * torch.ones(1, H, W) )
            Gs.append( pySim.getKernel_th( -td, mf, Nz, Nz, scatter_prop ).permute(2,0,1).unsqueeze(0) ) 

        I0 = torch.cat( I0, dim=0 )
        Gs = torch.cat( Gs, dim=0 ) 
        print('')

        # Images for gt Q #
        Q_gt = torch.zeros(Nz, Nr, Nc).cuda()
        Q_gt[ d_vein-1: d_vein+2, 50-1:52, : ] = 10 
        imgs_gt = pySim.Q_to_imgs(Q_gt, Gs.cuda(), I0.cuda(), approx_name = 'Raytov') 
        td_plot = torch.arange(0, 31, 1)
        vals_bkg = torch.zeros(len(td_plot), TD);
        val_vein = torch.mean(imgs_gt[:, 50, 50-30:50+30], dim =1).cpu() 
        for i in range( len(td_plot)):
            vals_bkg[i, :] = torch.mean(imgs_gt[:, 50+td_plot[i], 50-30: 50+30], dim = 1)
            vals_bkg[i, :] = vals_bkg[i, :] - val_vein

        # Get the gradient for each row of vals_bkg #
        grad_contrast = np.zeros((vals_bkg.shape[0], vals_bkg.shape[1])) 
        for i in range( len(td_plot)):
            grad_contrast[i, :] = np.gradient( vals_bkg[i, :].numpy() ) 
        array_grad_contrast.append( grad_contrast )

        # write the images #
        log_img = np.log( np.abs(grad_contrast) + .000001 )
        imshow( log_img,  cmap='jet', vmin= log_img.min(), vmax= log_img.max() )
        savefig('%03d.png'%(d))
        sio.savemat('figs/coeff_1_d_%d.mat'%(d), {'grad_contrast': grad_contrast})
        close()

if __name__ == '__main__':
    main()
