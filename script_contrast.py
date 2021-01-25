import numpy as np
import scipy.io as sio
import pyProcImg.procImg as procImg 
import pyUtils.mutils as pyutils 
import imageio

from matplotlib.pyplot import *

def main():

    r_vein = 1
    D_VEIN = [2, 6, 10]
    img_dt_paths = [ 'dat/img_dt_dvein_%d_rvein_%d.mat'%(d_vein, r_vein) for d_vein in D_VEIN ]
    # r_vein = 1
    # D_VEIN = [15, 20, 25, 30]
    # img_dt_paths = [ 'dat/img_dt_dvein_%d_rvein_%d_vertical.mat'%(d_vein, r_vein) for d_vein in D_VEIN ]
    # img_dt_paths = [ 'dat/img_dt_dvein_%d_rvein_%d_vertical.mat'%(d_vein, r_vein) for d_vein in D_VEIN ]

    BackRows = np.arange( 10,90 )
    for back_row in BackRows:
        print('back_row = %d'%(back_row)) 
        # figure(figsize=(16,16))
        figure()
        for img_dt_path in img_dt_paths:
            img_info = sio.loadmat(img_dt_path)
            IMGS_td = img_info['IMGS_td'] 
            dTs = img_info['dTs'].flatten()

            proflie_vein = np.mean(IMGS_td[50, 30:-30, : ], axis = 0)
            profile_back = np.mean(IMGS_td[back_row, 30:-30, : ], axis = 0)
            # proflie_vein = np.mean(IMGS_td[30:-30, 50, : ], axis = 0)
            # profile_back = np.mean(IMGS_td[30:-30, back_row, : ], axis = 0)

            # plot(proflie_back)
            # plot(proflie_vein)
            # plot(dTs, profile_back - proflie_vein, 'x-', label= img_dt_path) 
            plot(dTs, profile_back - proflie_vein,  label= img_dt_path) 

        legend(loc='upper right')
        title('back row = %d'%(back_row))
        xlabel('delta t')
        ylabel('intensity: back - vein')
        ylim([-10 , 10])
        grid()
        # show()
        savefig('diff_I_row_%d.png'%(back_row))
        close()

if __name__ == "__main__":
    main()
