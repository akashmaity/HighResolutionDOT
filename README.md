# highResolutionDOT

# Overview
This is an implementation of the following paper:

["High Resolution Diffuse Optical Tomography using Short Range Indirect Subsurface Imaging"](http://www.cs.cmu.edu/~ILIM/projects/LT/highRes_DOT/index.html), 
Chao Liu, Akash Maity, Artur W. Dubrawski, Ashutosh Sabharwal and Srinivasa G. Narasimhan.
IEEE International Conference on Computational Photography (ICCP) 2020

In this work, we present a fast imaging and algorithm for high resolution diffuse optical tomography with a line imaging and illumination system. Key to our approach is a convolution approximation of the forward heterogeneous scattering model that can be inverted to produce deeper than ever before structured beneath the surface.

# Getting Started:
1.	Download and install Mcxlab (http://mcx.space/wiki/index.cgi?Download#Download_the_Latest_Release)

2.	Add installed mcxlab and mcx path in `msetup.m` file:  
```
addpath(genpath('/path/to/mcxlab') )
```

# Generate Images

## Step 1.
In our simulation setup, we use scanning line source. The example code has 1 absorbing cylinders at a certain depth from the surface.
Generate images for each scanning position by running `mcx_synthesize_images.m`

## Step 2.
We will use the scanned images to generate “constant-delay images” or “Td” images by changing the filename in `script_get_images_td.py`, `line 26` and `line 29` :
```
path_rawimgs = '%s/%sN%d_unitmm_0.24_veins.mat'%(mcx_img_path, name_prefix, N)
mat_file = 'dat/img_dt_unitmm_0.24_veins.mat'
```
and then run:
```
python script_get_images_td.py
```

The images generate correspond to multiple source-to-detector pair distances respectively.

# Reconstruction

To generate 3D absorption reconstruction values, change the fname input in `line 35` to the generated Td images:
```
mat = sio.loadmat('%s/img_dt_unitmm_0.24_veins.mat'%(dat_path))
```
and then run:
```
python dipole_inverse_gpu.py
```
The optimized 3D absorption reconstruction is stored in `Q_opt_tdmin_5_Bonn_veins.mat`.
