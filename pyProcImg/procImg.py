
import os
import sys
import math
import numpy as np
import scipy.io as sio
import scipy.interpolate as sinterp
import scipy.misc as smisc
import PIL.Image as image
import PIL
import torchvision.transforms.functional as tv_func
import matplotlib.pyplot as plt 

import torch
import torch.nn.functional as F

import pyUtils.interp_irregular as interp_irr 

from matplotlib.pyplot import * 
def get_laser_profile_td( laser_profile, t_values, t_d, radius = None, 
                          kernel_size = None, fill_value = 0.):
    '''
    Get shifted laser profile, used for get g_l(delta_V)
    inputs:
            laser_profile - the laser profile, unit: pixel intensity value
            t_values - t values (x-axis of the laser profile) centered around zero, unit: pixel
            t_d - scalar, a.k.a v = t + delta_v, v is the projection row, t is the camera expsoure row 
            radius (optional) - radius of the laser spot
            kernel_size (optional) - the kernel size of the shifted 1d profile. If not set, then the 
                                     1D kernel size is len(laser_profile) 
    outputs:
            td_profile - shifted (move right-wards by t_d) 1D laser profile 
    '''
    assert len( laser_profile) == len(t_values)
    assert len( laser_profile) % 2 == 1 # odd number of points, s.t. center + two sides

    if kernel_size:
        assert kernel_size % 2 == 1 # odd number of points, s.t. center + two sides 

    r_max = radius

    if r_max is not None:
        assert r_max > 0
        laser_profile[t_values < -r_max] = 0 
        laser_profile[t_values > r_max] = 0 

    f_interp = sinterp.interp1d(t_values.flatten(), laser_profile.flatten(), 
            kind='quadratic', bounds_error=False, fill_value= fill_value)

    t_value_interp = t_values - t_d
    td_profile = f_interp(t_value_interp.flatten())

    if kernel_size: 
        mid = int( (len(laser_profile.flatten()) -1 ) /2 ) 
        kernel_r = int( (kernel_size -1) / 2)
        td_profile = td_profile[ mid- kernel_r: mid+ kernel_r+1]

    return td_profile
    
def synth_imgs_td_interp(IMGS_raw, Vs, t_range, delta_vs):
    '''
    synthesize multiple td images, each for one delta_v value in delta_vs 
    inputs: 
            IMGS_raw - nImg x H x W
            Vs -  array v's, corresponding to IMGS_raw, should be ascending order 
            t_range - (t_min, t_max), t_min/max are ints
            delta_vs - array of delta_vs, should be ascending order
    outputs: 
            img_tds - n_delta_vs x H x W tensor
    '''
    assert len(IMGS_raw) == len(Vs) 

    nimg_out = len(delta_vs)
    nimg_in, H, W = IMGS_raw.shape
    imgs_td = np.zeros((nimg_out, H, W)) 
    s_values = np.arange(0, W) # the colume coordiates
    v_values = Vs
    for t in range(t_range[0], t_range[1]+1):
        print('synthesizing row %d from %d to %d'%(t, t_range[0], t_range[1]))
        v_interp = t + delta_vs
        # get one row values in img_tds by interpolation # 
        vs_slice = IMGS_raw[:, t, :]
        fn_interp = sinterp.RectBivariateSpline(v_values, s_values, vs_slice) 
        vs_slice_interp = fn_interp(v_interp, s_values)
        imgs_td[:, t, :] = vs_slice_interp 

    return imgs_td 

def _meshGrid_loc( xv, yv):
    xx, yy = np.meshgrid(xv, yv)
    return np.vstack( (xx.flatten(), yy.flatten() ) )

def synth_imgs_td_irregular_interp(
        IMGS_rect, Vs_rect, 
        rotate_angles, delta_v, 
        t_range= None, if_debug = False, idx_debug = None, offset = .5):
    '''
    inputs: 
            IMGS_rect - nImg x H x W or a list with nImg elements
            Vs_rect -  array of v's, corresponding to IMGS_rect, should be ascending order 
            rotate_angles - array of rotation angles in degrees, using which to rotate image_raw to image_rect
            delta_v - int, delta_v = v_projct - t_line, the delay between projector and exposed camera line 
            t_range - [t_min, t_max] to draw
    outputs:
            img_td - The synthesized rolling shutter image, an image of the same size as IMGS_rect[0]
    '''

    use_gpu = True # use gpu for interpolation
    Vs_rect = Vs_rect - offset
    assert len(IMGS_rect) == len(Vs_rect) 
    H, W = IMGS_rect[0].shape
    nimg_in = len(IMGS_rect)

    points_loc = [] # data point locations in the raw image
    values = []  # values in the raw image 
    center_img = np.asarray( [float(W)/2, float(H)/2]) 
    idx_range = [idx_debug] if if_debug else range(nimg_in)

    # Get the irregular anchor points (points_loc, values_raw_img) from which we will interpolate #
    N = nimg_in
    imgs_tensor = torch.zeros(N, 1, H, W) 
    grid_out = torch.zeros(N, 1, W, 2) 
    for idx in idx_range: 
        imgs_tensor[idx, :, :, :] = torch.tensor( IMGS_rect[idx].astype(np.float) )
        v_rect = Vs_rect[idx]
        out_pos_x, out_pos_y = np.meshgrid( np.arange(0,W) + offset , v_rect - delta_v )

        # TODO: this is ok for even w and h, how about for odd w and h ?
        grid_out[idx, 0, :, 0] = torch.tensor((2*out_pos_x - W+1 ) / (W-1) )
        grid_out[idx, 0, :, 1] = torch.tensor((2*out_pos_y - H+1 ) / (H-1) )

        st_rect = np.zeros((2, W)) 
        st_rect[0,:] = np.arange(0, W) + offset # s, camera horizontal
        st_rect[1,:] = v_rect - delta_v     # t, camera vertical
        st_raw = rotate_pts( -rotate_angles[idx], st_rect, center_img )
        points_loc.append(st_raw)

    # do the interpolation #
    values_ = F.grid_sample(imgs_tensor.cuda(), grid_out.cuda() ).cpu().numpy() 
    values  = [ values_[indx, 0, 0, :] for indx in range(nimg_in) ] 

    # The second interpolation: Interpolate over irregular data points 
    # TODO: how can we speed this up ?
    #       Refer to splat in bilateral filter ?  
    points = np.hstack(points_loc).transpose()
    values = np.hstack(values).flatten() 
    s_values = np.arange(0, W) + offset
    t_values = np.arange(0, H) + offset
    s_values_draw = s_values
    t_values_draw = t_values if t_range is None else np.arange(t_range[0], t_range[1]) + offset
    st_grid_locs = _meshGrid_loc(s_values_draw, t_values_draw)

    xi = st_grid_locs.transpose()
    img_td = sinterp.griddata(points, values, xi, method='linear', fill_value=0.) 

    if t_range is not None:
        hh = int(t_range[1] - t_range[0])

    img_td = img_td.reshape((H,W)) if t_range is None else img_td.reshape((hh, W)) 

    #debug
    # grid_out_raw_x = ( (grid_out[:, :, :, 0] * W ) + W ) / 2
    # grid_out_raw_y = ( (grid_out[:, :, :, 1] * H ) + H ) / 2
    # values_map = values.reshape( (len(idx_range), W) )
    # map_v = values.reshape( (99, 100) )
    # return img_td, map_v 
    #end debug

    return img_td


def synth_imgs_td(IMGS_raw, Vs, t_range, delta_v):
    ''' 
    synthesize one td imgae 
    inputs: 
            IMGS_raw - list of input raw images, the projection lines are horizontal
            Vs - list of v's, corresponding to IMGS_raw
            t_range - (t_min, t_max), t_min/max are ints
            delta_v - int, v_projct - t_line, the delay between projector and exposed camera line 
    outputs: 
            img_td - an image of the same size as IMGS_raw[0]
    ''' 
    H,W = IMGS_raw[0].shape
    img_td = np.zeros((H,W), dtype = IMGS_raw[0].dtype)
    img_dv_err = np.zeros((H,W))
    nimgs = len(IMGS_raw)
    Vs_ = np.asarray(Vs) 
    err_dv_max = 0
    indx_img_max = 0
    t_max=0; v_max=0 

    for t in range(t_range[0], t_range[1]+1):
        # Pick the right raw image to copy from #
        v_proj = float(t) + float(delta_v) # the target projection line 
        # the difference between the actual projection line and the target projection line
        diff_vproj_Vs = Vs_ - v_proj 
        indx_img = np.argmin(np.abs(diff_vproj_Vs))
        err_dv = diff_vproj_Vs[indx_img] 
        # debug
#        if(err_dv > err_dv_max):
#            err_dv_max = err_dv
#            indx_img_max = indx_img
#            t_max = t
#            v_max = v_proj 
        # 
        img_proj = IMGS_raw[indx_img]
        # copy one row in one image of IMGS_raw, into img_td # 
        irow_cam = t
        img_td[irow_cam, :] = img_proj[irow_cam, :] 
        img_dv_err[irow_cam, :] = err_dv 

#    print('synth_imgs_td(): img_idx:%d, t:%d, delta_v:%d, v_expect:%d, v_proj:%f, err_dv:%f'%(\
#            indx_img_max, t_max, delta_v, v_max, Vs_[indx_img_max], err_dv_max) ) 

    return img_td, img_dv_err

def get2Dline_para(pt1,pt2, center=None):
    '''
    pt1, pt2 - in the format of (x,y)
    center - central coordiate to substract, such that the coordinate of the center of the image is (0, 0)
    ''' 
    if center is None:
        center = np.asarray([0.,0.])
    pt1_ = pt1 - center 
    pt2_ = pt2 - center 
    A = np.zeros((2,2))
    A[0,:] = pt1_
    A[1,:] = pt2_
    line_para = np.linalg.pinv(A).dot(np.ones((2,1)) )
    return line_para 

def rotate_img(img_in, rot_r, center=None):
    '''
    rotate an image
    inputs: img_in - A PIL image to rotate
            rot_r -  In degrees degrees counter clockwise order
            center (2-tuple, optional) - Optional center of rotation
                                         Origin is the upper left corner. Default is the center of the image
    output: img_out- a PIL image, rotated
    ''' 
    img_out = tv_func.rotate(img_in, rot_r, PIL.Image.BILINEAR, expand=False, center=center)
    return img_out 

def rotate_pts(norm_angle, pts, center_img):
    '''
    inputs:
    norm_angle - rotation angle in degrees, anti-clockwise
    pts - 2 x npts, each col is one point
    center_img - 2 x 1, the center of the image

    outputs:
    pts_rotated
    '''
    center_img_ = center_img.reshape(2,1)
    rotM = np.zeros((2,2))
    rotM[0,0] = rotM[1,1] = math.cos(norm_angle*math.pi/180.)
    rotM[0,1] = -math.sin(norm_angle*math.pi/180)
    rotM[1,0] =  math.sin(norm_angle*math.pi/180)
    pts_rotated = rotM.dot( pts - center_img_) + center_img_
    return pts_rotated

def toTensor(img_in):
    '''
    convert np array or PIL image to tensor
    '''
    img = tv_func.to_tensor(img_in) 
    return img

def read_img( fname, return_pil=False ):
    '''
    read image
    input : fname - file path
    output: img - a np array, BGR channels
    '''
    img = image.open(fname)
    if not return_pil:
        img = np.asarray(img).astype(np.float32) 
    return img 

def add_imgs(imgs, twin_r, ):
    '''
    sum up and return multiple images, centered idx, with radiu =  twin_r

    input: 
    imgs - a 3D volume of gray scale images size [H, W, D]
    twin_r - time window radius, 

    output: 
    imgs_sum - a 3D volume of the same size as imgs. 
    imgs_sum[:, :, idx] = sum( imgs[:, :, idx-twin_r:idx+twin_r+1], axis=2)
    '''
    H, W, D = imgs.shape
    imgs_sum = np.zeros( (H, W, D) , dtype= imgs.dtype); 
    for idx in range( twin_r, D-twin_r):
        sys.stdout.write("\r" + "image %d/%d"%(idx, D-twin_r) )
        imgs_sum[:, :, idx] = np.sum( imgs[:, :, idx-twin_r: idx+twin_r+1], axis=2) 
    print('\n')
    return imgs_sum 

def write_imgs(imgs, fpath, vmax):
    '''
    write a set of image to folder path
    '''
    if not os.path.exists(fpath):
        os.mkdir(fpath) 

    H, W, D = imgs.shape 
    for ifile in range( D):
        sys.stdout.write("\r" + "writing image %d/%d"%(ifile, D) )
        file_path = '%s/img_%06d.png'%(fpath, ifile)
        img_arr = imgs[:, :, ifile].astype(np.float32) / vmax
        smisc.imsave( file_path, img_arr) 
