import sys
import os 
import scipy.misc as smisc
import numpy as np
import torch
import scipy.io as sio

import torchvision.transforms as vision_F
import torch.nn.functional as F

from matplotlib.pyplot import *

def depthMap(Q_opt):
    '''Q_opt - D x H x W
    '''
    D, H, W = Q_opt.shape
    Q_opt[Q_opt< 0 ] =0

    dsum= np.zeros( (H,W) )
    qsum= np.zeros( (H,W) )

    for d in range(0, D):
        dsum  = dsum + d * Q_opt[d]         
        qsum = qsum + Q_opt[d]

    dmap = dsum / (qsum+10e-8)
    
    return dmap

def subsample_1d(signal, rate):
    '''
    inputs:
    signal - input 1d signal, size: 1 x 1 x length
    rate - subsample rate. output size = input size / rate
    '''

    assert signal.device.type == 'cpu'
    assert signal.numel() %2 ==1 # odd number of elements in signal

    # 1D binning beofre interpoloation #
    # kernel = torch.ones( ( 1,1, int(rate) ) )/ rate
    # signal_conv = F.conv1d( signal, kernel, padding= int(rate)//2 )
    signal_conv = signal

    # 1D interpolation #
    x_in = torch.linspace( 1, signal.numel(), signal.numel() )
    x_in = x_in - x_in[len(x_in)//2]
    x_in_r = x_in[len(x_in)//2 +1 : ]
    x_in_l = x_in[0: len(x_in)//2]

    x_out_l = torch.linspace( x_in_l[0], x_in_l[-1], x_in_l.numel() // rate )
    x_out_r = torch.linspace( x_in_r[0], x_in_r[-1], x_in_r.numel() // rate )

    y_out_l = np.interp( x_out_l.numpy(), x_in.numpy(), signal_conv.numpy().flatten() )
    y_out_r = np.interp( x_out_r.numpy(), x_in.numpy(), signal_conv.numpy().flatten() )

    signal_sub = torch.zeros( 1+ len(y_out_l) + len(y_out_r)  )

    signal_sub[ 0: len(signal_sub)//2] = torch.FloatTensor(y_out_l)
    signal_sub[ len(signal_sub)//2 ] = signal_conv[ ..., signal_conv.numel()//2 ]
    signal_sub[ len(signal_sub)//2+1 : ] = torch.FloatTensor(y_out_r)

    return signal_sub.unsqueeze(0).unsqueeze(0)

def subsample_2d(imgs,  rate, ):
    '''
    input:
    imgs - 3D input data of size nchannel x H x W 
    rate - subsample rate. output size = input size / rate
    if_avg - if take the average over a n x n kernel ( n=3 for rate =2 ; n=5 for rate=4)
    '''
    assert imgs.device.type == 'cpu'

    C, H, W = imgs.shape 
    subH, subW = H//rate, W//rate 
    resizer          = vision_F.Resize((subH, subW), interpolation= 2)
    converter2pil    = vision_F.ToPILImage() 
    converter2tensor = vision_F.ToTensor()

    # average over kernels (optional) #

    # subsample image based on 2d interpolation #
    max_v = imgs.max()
    imgs_sub = [] 
    for img in imgs:
        img_pil = converter2pil( img / max_v ) 
        img_pil_resize = resizer( img_pil ) 
        imgs_sub.append( converter2tensor( img_pil_resize ) ) 

    imgs_sub = torch.cat( imgs_sub, dim=0 ) * max_v

    return imgs_sub

def subsample_2d_pooling(imgs, rate):
    '''
    do the subsampling through average pooling
    input:
    imgs - 3D input data of size nchannel x H x W 
    rate - subsample rate. output size = input size / rate
    ''' 
    C, H, W = imgs.shape
    kernel_radius = rate-1
    kernel_size = 2 * (rate-1) + 1
    padding = kernel_radius
    imgs_sub = F.avg_pool2d( imgs, kernel_size= kernel_size,
                             stride = rate, padding = padding,
                             count_include_pad = False )

    return imgs_sub

def get_tdmask(Tds, td_range):
    td_min, td_max  = td_range[0], td_range[1] 
    Td_mask = torch.zeros(len(Tds))
    abs_tds = torch.abs( Tds )
    Td_mask[ (abs_tds>=td_min) * (abs_tds <= td_max) ] =1.  

    return Td_mask 

def m_makedir(dirpath):
    '''
        make dir
        dirpath -
    '''
    if not os.path.exists(dirpath):
        os.makedirs( dirpath) 

def write_imgs(imgs, fpath, vmax):
    '''
    write a set of image to folder path
    input:
        imgs - imgs[:, :, img]
        fpath - the folder path to save the images
        vmax - the max value for the dynamic range
    '''

    if not os.path.exists(fpath):
        os.makedirs(fpath) 
    H, W, D = imgs.shape 

    for ifile in range( D):
        sys.stdout.write("\r" + "writing image %d/%d"%(ifile, D) )
        file_path = '%s/img_%06d.png'%(fpath, ifile)
        img_arr = imgs[:, :, ifile].astype(np.float32) / vmax
        smisc.imsave( file_path, img_arr) 

def write_figures(imgs, fldr_path, vrange, prefix=None, cmap='jet'):
    '''
    write images as matplotlib figures
    inputs:
        imgs      - imgs[i_img, :, :]
        fldr_path - the folder path to save the images 
        vrange - [vmin, vmax]
    '''
    for idx, img in enumerate(imgs):
        # print('idx=%d'%(idx))
        fname = '%04d.png'%(idx) if prefix is None else '%s%04d.png'%(prefix, idx)
        imsave( '%s/%s'%(fldr_path, fname), img,
                vmin=vrange[0], vmax=vrange[1], cmap = cmap)


def showSym(imgs, indx, indx_center):
    '''
    show imgs[ indx_center - indx ] - imgs[ indx_center + indx ] 
    '''
    i = indx_center
    j = indx
    imshow((imgs[i -j ][:, :, 1].astype('float32') - imgs[i + j ][:, :, 1].astype('float32')), 
            vmin=-30, vmax=30, cmap='jet'); 
    title('offset %d'%(j)) 

    return

