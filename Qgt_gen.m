N = 256;
Nr = 256;
Nc = 256;
Nz = 256;

% if_vertical = 0;
% load('prop_vol_scale_3.mat')
scale = 3;%4.1667; %3; % 10

unitinmm = 1./scale;
z_surf = 0;
depth = 4;
if_vertical = 0;
d_vein = depth/unitinmm;
r_vein = 1/unitinmm;

z_vein = d_vein+z_surf;
Q_gt=(zeros(N,N,N));


Q_gt(round(Nr/2-10/unitinmm-r_vein-1):round(Nr/2-10/unitinmm+r_vein+1),1:Nc, round(z_vein-r_vein-1):round(z_vein+r_vein+1)) = 2.3;

Q_gt(round(Nr/2+10/unitinmm-r_vein-2):round(Nr/2+10/unitinmm+r_vein+2),1:Nc, round(z_vein-r_vein-2):round(z_vein+r_vein+2)) = 2.3;

Q_gt(round(Nr/2-r_vein):round(Nr/2+r_vein+1),1:Nc, round(z_vein-r_vein-1):round(z_vein+r_vein+1)) = 2.3;

save('Q_gt_3rods_scale_3_depth4.mat','Q_gt');