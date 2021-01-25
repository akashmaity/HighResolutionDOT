# M2 : dipole model for inversing the image #
import torch
import numpy as np
import math
import scipy.io as sio
import pySim.image_amp as pySim
from matplotlib.pyplot import * 

# TODO: 
# Global variables #
#model_name = 'Bonn'

g = 0.9 
mua =  0.05 # 0.045/4
musp = 18 * (1-g) #35.65 * (1-g) #10 * (1-g)
D = 1/(3*(mua+musp)) 
beta = math.sqrt( 3*musp*mua )
Reff = 0.496 

# Multiplication factor 1 px = mf mm #
mf_ = 0.50;
mf = [mf_,mf_, mf_] # [mf_uv, mf_st, mf_z]
Nxy = 65
Nz = int( 8 / mf[2])

    
def script_opt_Q( d_vein, td_mask_min, max_iter, model_name, use_diffimg, vein_radius = 1, dat_path= ''):
    scatter_prop = {'g':g, 'mua':mua, 'musp':musp, 'D':D, 'Reff':Reff, 'beta':beta} 
    d_vein= d_vein
    max_iter , td_min = max_iter, td_mask_min 
    mat = sio.loadmat('%s/img_dt_unitmm_%0.2f_ppg_tumor_sample3.mat'%(dat_path,mf_))

    
    # debug: load the homogenous data
    # mat = sio.loadmat('./dat/img_dt_homogenous.mat') 
    # use GT optical properties #
    res_name = 'dat/Q_opt_tdmin_%d_%s_%0.2f_ppg_tumor_sample3.mat'%(td_min, model_name,mf_) # file to save the results 
    Imgs, Td = mat['A'], mat['dTs'].flatten() 
    R = torch.FloatTensor(mat['M'])


    # Imgs = Imgs / 100. # cm^2 to mm^2 
    TD = len(Td) 
    N = Imgs.shape[0]
    Nr, Nc = N, N
    
    # debug: mask out boundary
    Imgs[ :10, : , :] = 0
    Imgs[ -10:,: , :] = 0
    Imgs[ :, :10 , :] = 0
    Imgs[ :, -10:, :] = 0 

    # Optimizing || I_m - I0 (1 - Q \conv gs) ||2 + other constriants #
    Imgs = torch.FloatTensor( Imgs ).permute(2, 0, 1)
    n_delay, H, W = Imgs.shape[0], Imgs.shape[1], Imgs.shape[2] 

    # plot( Td, (val_bkg - val_vein ).numpy(),'x-') ; grid(); xlabel('td'); title( 'val_{bkg} - val_{vein}' ); 
    # get Gs: n_delay x D x w x w
    # get I0: n_delay x H x W

    I0, I0_ref, Gs = [], [], []
    props = torch.FloatTensor( [ D, beta, Reff] ) 
    for td in Td:
        print( 'getting I0 and Gs for td=%f (Td_max= %f)'%( td, Td.max() ), end='\r' )
        # I0.append( pySim.getIntensity( -td, Nz, mf, scatter_prop) * torch.ones(1, H, W) ) #
        #I0_ = pySim.getIntensity( -td, Nr, mf, scatter_prop)
        
        I0_ref_ = pySim.getIntensity_th(-td, Nr, mf, props) 
        #I0.append(I0_)
        I0_ref.append(I0_ref_)
        Gs.append( pySim.getKernel_th( -td, mf, Nxy, Nz, scatter_prop ).permute(2,0,1).unsqueeze(0) ) 
        #Gs.append(kernel[35:66,35:66,:,td])
    sio.savemat('test.mat',{'I0':np.array(I0_ref)})
    #import pdb
    #pdb.set_trace()
    I0 = torch.FloatTensor( I0 ).unsqueeze(1).unsqueeze(2)
    I0_ref = torch.FloatTensor( I0_ref ).unsqueeze(1).unsqueeze(2)

    Gs = torch.cat( Gs, dim=0 ) 
    print('')

    # images for gt Q #
    Q_gt = 0.01*torch.ones(Nz, Nr, Nc).cuda()
    Q_gt[0:5,:,:] = 0.005
    Q_gt[5:,:,:] = 0.01


    imgs_gt = pySim.Q_to_imgs(Q_gt, Gs.cuda(), I0_ref.cuda(), R.cuda(), approx = model_name)

    # define Td_mask # 
    Q_init = None #None #Q_gt 
    # debug: use Q_gt #
    #Q_init = Q_gt 

    
    Gs_in = Gs
    Imgs_in = Imgs

    mask = None

    td_max = 32
    Td_mask = torch.zeros(len(Td)) if td_max is not None else torch.ones( len(Td) ) 
    if td_max is not None:
        Td_mask[ len(Td)//2-td_max: len(Td)//2+td_max+1  ] = 1. # assuming tds are centered around 0 

    Td_mask[ len(Td)//2-td_min : len(Td)//2+td_min+1 ] = 0.
    Td_mask = Td_mask.cuda()
        # if td_min > 0:
        #     Td_mask = torch.ones( len(Td) ) 
        #     Td_mask[ len(Td)//2-td_min : len(Td)//2+td_min+1 ] = 0.  
        # Td_mask = Td_mask.cuda()
    I0_ref_in = I0_ref

    ## ADAM ##
    # for Bonn diff
#     Q_opt = pySim.opt_Q( Imgs_in.cuda(), Gs_in.cuda(), I0_ref_in.cuda(), 
#                          step = .0002, max_iter = max_iter, opt_name = 'Adam', 
#                          Q_init = Q_init, lambdas= [.0001/10, .002/2], Td_mask = Td_mask,
#                          model = model_name, use_diffimg = use_diffimg, mask = mask )
    #sparse : .0001/50

    if(model_name =='Bonn'):
        Q_opt = pySim.opt_Q( Imgs_in.cuda(), Gs_in.cuda(), I0_ref.cuda(), R.cuda(),
                          step = .001, max_iter = max_iter, opt_name = 'Adam', 
                          Q_init = Q_init, lambdas= [.001, .001, 0.0], Td_mask = Td_mask, model = model_name, use_diffimg = use_diffimg )
    else:  
        # for Raytov diff log scale
        Q_opt = pySim.opt_Q( Imgs_in.cuda(), Gs_in.cuda(), I0_ref_in.cuda(), R.cuda(),
                             step = .001, max_iter = max_iter, opt_name = 'Adam', 
                             #Q_init = Q_init, lambdas= [0.05, .01, 0.00005], Td_mask = Td_mask,
                             Q_init = Q_init, lambdas= [5000, 1, 50], Td_mask = Td_mask,
                             model = model_name, use_diffimg = use_diffimg, mask = mask )

    
     
    print('') 
    imgs_opt = pySim.Q_to_imgs(Q_opt, Gs.cuda(), I0_ref.cuda(), R.cuda(), approx = model_name) 
    #diff_gt = imgs_gt.cpu() - Imgs.cpu()
    diff_opt = imgs_opt - Imgs.cuda()
    imgs_opt = imgs_opt.squeeze().detach().cpu()
    imgs_gt = imgs_gt.squeeze().detach().cpu()
    Q_opt = Q_opt.detach().cpu().numpy()
    Q_gt = Q_gt.detach().cpu().numpy()

    #for i in range( diff_gt.shape[0] ):
        #print('i=%d: diff_gt = %.3f \t diff_opt = %.3f'%(i, torch.norm( diff_gt[i,...]) * 1000, torch.norm( diff_opt[i,...] ) * 1000 )) 

    sio.savemat( res_name, {'Q_opt': Q_opt,'Q_gt': Q_gt, 'imgs_opt':imgs_opt.numpy(),'imgs_gt':imgs_gt.numpy(), 'Imgs': Imgs.cpu().numpy(), 'Gs_model': Gs_in.cpu().numpy(),'I0_model': I0_ref_in.cpu().numpy()}  ) 
    print('results saved to %s'%( res_name ) )

if __name__ == '__main__':
    
    model_name = 'Raytov'
    D_vein = [3.0]
    
    max_iter = 1000
    # max_iter = 500
    vein_radius = 1
    # dat_path = '/home/chaoliu1/'
    dat_path = 'dat'
    
    for d_vein in D_vein:
        if(model_name=='Bonn'):
            use_diffimg = False
            td_mask_min = 5
            script_opt_Q( d_vein,  td_mask_min, max_iter, model_name, use_diffimg, vein_radius = vein_radius, dat_path = dat_path )
        else:
            use_diffimg = True
            td_mask_min = 5
            script_opt_Q( d_vein,  td_mask_min, max_iter, model_name,use_diffimg, vein_radius = vein_radius , dat_path = dat_path )

    
