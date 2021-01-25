% The projection lines are slit patterns


clear all;
close all;
msetup;

%---------Scene description  ----------%
% % N = 256;
% % Nr = 256;
% % Nc = 256;
% % Nz = 256;

% if_vertical = 0;
load('prop_vol_scale_3.mat')
scale = 3;%4.1667; %3; % 10

unitinmm = 1./scale;
% z_surf = 0;
% depth = 3;
% if_vertical = 0;
% d_vein = depth/unitinmm;
% r_vein = 1/unitinmm;

% z_vein = d_vein+z_surf;
%cfg.vol=uint8(ones(N,N,N));

% if if_vertical == 0
%     str_vein1    = sprintf('{"Cylinder": {"Tag":2, "C0": [%f, 0, %d], "C1": [%f, %d, %d], "R": %d}},', ...
%         Nr/2-10/unitinmm, z_vein, Nr/2-10/unitinmm, Nc, z_vein, r_vein+1 );
%     str_vein2    = sprintf('{"Cylinder": {"Tag":2, "C0": [%f, 0, %d], "C1": [%f, %d, %d], "R": %d}},', ...
%         Nr/2, z_vein, Nr/2, Nc, z_vein, r_vein );
%     str_vein3    = sprintf('{"Cylinder": {"Tag":2, "C0": [%f, 0, %d], "C1": [%f, %d, %d], "R": %d}}]}', ...
%         Nr/2+10/unitinmm, z_vein, Nr/2+10/unitinmm, Nc, z_vein, r_vein+2 );
% else
%     str_vein    = sprintf('{"Cylinder": {"Tag":2, "C0": [0, %f, %d], "C1": [%d, %f, %d], "R": %d}}]}', ...
%         Nc/2 , z_vein, Nr, Nc/2, z_vein, r_vein );
%     
% end
% str_zlayers = sprintf('{"Shapes":[ {"ZLayers":[[0,%d, 1]]},', Nz);
% cfg.shapes=[str_zlayers str_vein1 str_vein2 str_vein3 ] ;



temp = prop_vol(1,:,:,12:25);
prop_vol(1,:,:,:) = 0.0458/4;
prop_vol(1,:,:,1:14) = temp;
prop_edit = squeeze(prop_vol(1,:,:,:));
%prop_edit(prop_edit>0.0458/4) = 10*prop_edit(prop_edit>0.0458/4);
prop_vol(1,:,:,:)=prop_edit;

cfg.vol = single(prop_vol);

src_r = 2:0.5:6;
src_c = 2:0.5:6;
[sr,sc] = ndgrid(src_r,src_c);
srcs = [sr(:),sc(:)]*32;

det_r = 1:0.5:7;
det_c = 1:0.5:7;
[dr,dc] = ndgrid(det_r,det_c);
dets = [dr(:),dc(:)]*32;

is_individual = 1;

no_dts = length(dets);
no_srcs = length(srcs);



i = 0;
for s = 1 : no_srcs

% 
prop=[0.0000    0.0  1.0000    1; ...
          0.01145  10   0.9000    1.3700; ...
           5      10    .9       1.37 ];
       
    cfg.unitinmm = unitinmm;   
    cfg.prop = prop;
    cfg.gpuid= 3;

    cfg.srcpos =[srcs(s,1) srcs(s,2) 0];
    cfg.srctype = 'pencil';
    cfg.nphoton=2e7;
    cfg.isreflect=0;
    cfg.issrcfrom0=0;
    cfg.tstart=0;
    cfg.tend=1e-6;
    cfg.tstep=1e-6;
    cfg.srcdir=[0 0 1];
    cfg.autopilot=1;
    cfg.debuglevel='P';
    
    [flux]=mcxlab(cfg);
    
    flx=flux.data;
    dflux =flx(:,:,1)*cfg.tstep;
    
%     for d = 1 : 4
%         Images(:,:,s) = dflux;
%         r = dets(d,1);
%         c = dets(d,2);
%         Det_flux_mat(d,s) = dflux(r,c);
    if(is_individual)
          Det_flux_mat(1,s) = dflux(srcs(s,1)-32,srcs(s,2)-32);
          Det_flux_mat(2,s) = dflux(srcs(s,1)-32,srcs(s,2)+32);
          Det_flux_mat(3,s) = dflux(srcs(s,1)+32,srcs(s,2)+32);
          Det_flux_mat(4,s) = dflux(srcs(s,1)+32,srcs(s,2)-32);
          
          Det_loc_mat{1,s} = [srcs(s,1)-32,srcs(s,2)-32];
          Det_loc_mat{2,s} = [srcs(s,1)-32,srcs(s,2)+32];
          Det_loc_mat{3,s} = [srcs(s,1)+32,srcs(s,2)+32];
          Det_loc_mat{4,s} = [srcs(s,1)+32,srcs(s,2)-32];
    else
        for d = 1 : no_dts
            Images(:,:,s) = dflux;
            r = dets(d,1);
            c = dets(d,2);
            Det_flux_mat(d,s) = dflux(r,c);
        end
    end
    
end

if(is_individual)
    Det_flux = Det_flux_mat(:);
else
    Det_flux = sum(Det_flux_mat,2);
end
save('sparse_DOT','Det_flux');
% imagesc(log10(sum(Images,3)));

