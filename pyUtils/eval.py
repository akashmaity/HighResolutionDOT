import sys
import os
import scipy.misc as smisc
import numpy as np
import torch
import scipy.io as sio

import torchvision.transforms as vision_F
import torch.nn.functional as F
from scipy.ndimage import zoom


from matplotlib.pyplot import *


def _get_IoU( img, img_ref ):

    assert img.max()<=1 and img_ref.max()<=1
    mask_intersect = ( (img + img_ref) ==2)
    # import ipdb; ipdb.set_trace()
    mask_envelope = ((img>0).astype( np.float32 ) + (img_ref>0).astype( np.float32 )) >0
    # mask_envelope = (img>0).astype( np.float32 )
    mask_intersect = mask_intersect.astype( np.float32 )
    mask_envelope = mask_envelope.astype( np.float32 )
    return mask_intersect.sum() / mask_envelope.sum()



def IoU_2D(Q_opt, Q_gt, thresh_range, mask = None):
    '''
    from Q_opt and Q_gt, get IoU in 2D
    '''

    D, H, W = Q_opt.shape
    if mask is None:
        mask = np.ones( (H, W) )
    q_opt_max = Q_opt.max(axis=0)
    q_gt_max = Q_gt.max(axis=0)

    iou_best = -1;
    mask_best = None
    # import pdb; pdb.set_trace()
    for thresh in np.linspace( thresh_range[0], thresh_range[1], 1000 ):
        q_opt_mask = q_opt_max > thresh
        q_gt_mask = q_gt_max > 0.5

        q_opt_mask = q_opt_mask.astype( np.float32 )
        q_gt_mask = q_gt_mask.astype(np.float32)

        iou = _get_IoU( q_opt_mask* mask, q_gt_mask* mask)
        if(iou> iou_best):
            iou_best = iou
            mask_best = q_opt_mask * mask

    return iou_best, mask_best, q_gt_mask * mask

def eval_gsnxy():
    import os.path
    nxy_range = np.arange(11, 71, 2)
    thresh_range = [0.01, 5]
    #mask = sio.loadmat('../tmp_res/costMap_mask.mat')['maskI'][0].astype(np.float32)
    #mat_gt = sio.loadmat('../pySim/vein_tumor_prop_vol_scale_3.mat')

    IOU = []
    for nxy in nxy_range:
        print(nxy)
        # mat = sio.loadmat('../tmp_res/opt_Q_opt_laser_profile_vein_tumor_2_scale_3_marble_gs_nxy_%d_maxiter_300.mat'%(nxy))
        # '../tmp_res/opt_Q_opt_laser_profile_vein_tumor_2_scale_3_milk_gs_nxy_11_maxiter_300.mat'

        fname = '../tmp_res/opt_Q_opt_laser_profile_vein_tumor_2_scale_3_gs_nxy_%d.mat'%(nxy)
        if not os.path.exists(fname):
            fname = '../tmp_res/opt_Q_opt_laser_profile_vein_tumor_2_scale_3_gs_nxy_%d_maxiter_300.mat'%(nxy)

        mat = sio.loadmat(fname)
        Q_opt = mat['Q_opt']
        prop_vol = mat_gt['prop_vol']
        sigma_s = prop_vol[0]
        Q_gt = sigma_s
        Q_gt = np.transpose(Q_gt, (2, 0, 1) )
        iou, mask_best, mask_gt = IoU_2D( Q_opt, Q_gt , thresh_range, mask=None)
        IOU.append(iou)

        # import ipdb; ipdb.set_trace()

    IOU = np.asarray(IOU)
    import ipdb
    ipdb.set_trace()

def eval_mus():

    thresh_range = [0.1, 30]
    #mask = sio.loadmat('../costMap_mask.mat')['maskI'][0].astype(np.float32)
    mat_gt = sio.loadmat('../Q_gt_3rods_scale_3_depth3.mat')
    MUS = [10]
    DEPTH = [1,2,3,4]

    IOU = []
    for mus in MUS:

        # fname = '../tmp_res/opt_Q_opt_laser_profile_vein_scale_3_skin_gs_nxy_33_depth_%d.mat'%(depth)
        fname = '../inverseDOT_mus10.mat'
        mat = sio.loadmat(fname)
        Q_opt = mat['Q']
        Q_opt = zoom(Q_opt, (4, 4, 4))
        #prop_vol = mat_gt['prop_vol']
        Q_gt = mat_gt['Q_gt']
        Q_gt = Q_gt[:,:,0:32]
        sigma_s = mus
        #Q_gt = sigma_s
        Q_gt = np.transpose(Q_gt, (2, 0, 1) )
        Q_opt = np.transpose(Q_opt, (2, 0, 1) )
        #import pdb
        #pdb.set_trace()
        iou, mask_best, mask_gt = IoU_2D( Q_opt, Q_gt , thresh_range, mask=None)
        IOU.append(iou)

        # import ipdb; ipdb.set_trace()

    IOU= np.asarray(IOU)
    import pdb
    pdb.set_trace()
    
    
def variance_depth():
    '''evaluate the contrast'''
    nxy = 33
    mask = sio.loadmat('../tmp_res/costMap_mask.mat')['maskI'][0].astype(np.float32)
    DEPTH = [1,2,3,4]
    VAR = []
    for depth in DEPTH:
        print(nxy)
        fname = '../inverseDOT_mus_%d.mat'%(depth)
        img = sio.loadmat(fname)['IMGS_td']
        img = img/img.max()
        img_sum = img[:,:,20:35].sum(axis=2)
        img_sum_v = img_sum.flatten()[mask.flatten()>0]
        var = np.var(img_sum_v )
        VAR.append(var)

    VAR = np.asarray(VAR)
    import ipdb
    ipdb.set_trace()



if __name__ == '__main__':
    eval_mus()
    # variance_depth()