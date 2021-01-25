import numpy as np
import sys
import cv2
import scipy.io as sio
# import matplotlib.pyplot as plt 
import pyProcImg.procImg as procImg 
import pyUtils.mutils as pyutils 
import imageio
from matplotlib.pyplot import * 

def render( name_prefix,   max_dt, N = 300, dt_step=1, if_vertical=False, mcx_img_path = 'dat/mcx_imgs' ):
    # scene setup
    #d_vein = v_depth
    #r_vein = v_radius
    # syntheisze parameters
    dt_min, dt_max = -max_dt, max_dt
    
    write_imgs = False
    write_mat = True

    if not if_vertical:
        #path_rawimgs = './dat/mcx_imgs/mcx_imgs_N%d_vd_%.2f_vr_%.2f.mat'%(N, d_vein, r_vein )
        #path_rawimgs = './dat/mcx_imgs/mcx_imgs_cube_vein_N%d_vd_%.2f_vr_%.2f.mat'%(N, d_vein, r_vein )

        # Input file name
        path_rawimgs = '%s/%sN%d_unitmm_0.24_veins.mat'%(mcx_img_path, name_prefix, N)

        # Output file name
        mat_file = 'dat/img_dt_unitmm_0.24_veins.mat'

        # debug: generate the homogenous td images #
        # path_rawimgs = 'dat/mcx_imgs_N100_vd_10.00_vr_3.00_homogenous.mat'
    else:
        # path_rawimgs = '%s/%sN%d_vd_%.2f_vr_%.2f_vertical.mat'%(mcx_img_path, name_prefix, N, d_vein, r_vein )
        path_rawimgs = 'dat/mcx_imgs_N256_unitmm_0.24_vdepth.mat'
        mat_file = 'dat/img_dt_dvein_%.1f_rvein_%d_vertical.mat'%( d_vein, r_vein )
    print('path_rawimgs: %s\n'%(path_rawimgs))

    # path_rawimgs = './mcx_imgs_N200.mat'
    img_info = sio.loadmat( path_rawimgs )
    Imgs = img_info['Imgs']
    Vs   = img_info['Lx']
    
    IMGS_rect = np.transpose( Imgs, [2, 0, 1] )
    IMGS_rect[IMGS_rect > 255] = 255

    # TODO: here .5 is needed to get symetrical td images for simulation. How about for real images ?
    vs_rect = Vs.flatten()
    rotate_angles = np.zeros( (IMGS_rect.shape[0], 1) ) 

    dTs = np.arange(dt_min, dt_max+1,  dt_step ) 
    t_range = None
    
    IMGS_td = []
    # maps_v = []
    for i, dt in enumerate( dTs ):
        print('synthesizing %d/%d images'%( i+1, len(dTs) ),end='\n' )

        img_td = procImg.synth_imgs_td_irregular_interp(IMGS_rect, vs_rect.astype(np.double), 
                                                        rotate_angles, delta_v= dt, t_range = t_range, offset=0.5)         # maps_v.append(map_v)
        IMGS_td.append(img_td)

    IMGS_td = np.dstack( IMGS_td ) 

    if write_mat: 

#        sio.savemat( mat_file, 
#                     {'IMGS_td': IMGS_td, 'd_vein': img_info['d_vein'], 
#                      'r_vein': img_info['r_vein'], 'prop': img_info['prop'],
#                      'Lx': vs_rect, 'dTs': dTs} )
        sio.savemat( mat_file, 
                     {'IMGS_td': IMGS_td, 'd_vein': 1, 
                      'r_vein': 1, 'prop': img_info['prop'],
                      'Lx': vs_rect, 'dTs': dTs} )
        print('saved mat : %s\n'%(mat_file ))

if __name__ == "__main__":

    N = 256 
    max_dt = 50 # dt range in [-max_dt, max_dt] 

    # mcx_img_path = './dat/skin_red/mcx_imgs'
    mcx_img_path = './dat'
    name_prefix = 'mcx_imgs_'

    if_vertical = False 

    
    render(name_prefix, N = N, max_dt = max_dt, if_vertical = if_vertical , mcx_img_path = mcx_img_path); 

    print('done')