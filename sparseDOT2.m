clear all;
close all;
msetup;

%---------Scene description  ----------%

% if_vertical = 0;
load('prop_vol_scale_3.mat')
scale = 3;%4.1667; %3; % 10

unitinmm = 1./scale;




temp = prop_vol(1,:,:,12:25);
prop_vol(1,:,:,:) = 0.0458/4;
prop_vol(1,:,:,1:14) = temp;
prop_edit = squeeze(prop_vol(1,:,:,:));
prop_edit(prop_edit>0.0458/4) = 100*prop_edit(prop_edit>0.0458/4);
%prop_edit(prop_edit>0.0458/4) = 0.0458/4;
prop_vol(1,:,:,:)=prop_edit;
vol1 = squeeze(prop_vol(1,:,:,:));
vol2 = squeeze(prop_vol(2,:,:,:));

volr1 = imresize3(vol1,0.5,'linear');
volr2 = imresize3(vol2,0.5,'linear');
vol_data(1,:,:,:) = volr1;
vol_data(2,:,:,:) = volr2;
cfg.vol = single(vol_data);

src_r = 2:2:10;
src_c = 2:2:10;
[sr,sc] = ndgrid(src_r,src_c);
srcs = [sr(:),sc(:)]*10;

det_r = 1:2:11;
det_c = 1:2:11;
[dr,dc] = ndgrid(det_r,det_c);
dets = [dr(:),dc(:)]*10;

is_individual = 1;

no_dts = length(dets);
no_srcs = length(srcs);



i = 0;
for s = 1 : no_srcs
    s
    cfg.gpuid= 5 ;

% 
prop=[0.0000    0.0  1.0000    1; ...
          0.01145  10   0.9000    1.3700; ...
           50      10    .9       1.37 ];
       
    cfg.unitinmm = unitinmm;   
    cfg.prop = prop;
    cfg.gpuid= 3;

    cfg.srcpos =[srcs(s,1) srcs(s,2) 0];
    cfg.srctype = 'pencil';
    cfg.nphoton=2e7;
    cfg.isreflect=0;
    cfg.issrcfrom0=0;
    cfg.tstart=0;
    cfg.tend=1e-7;
    cfg.tstep=1e-7;
    cfg.srcdir=[0 0 1];
    cfg.autopilot=1;
    cfg.debuglevel='P';
    
    [flux]=mcxlab(cfg);
    
    flx=flux.data;
    dflux =flx(:,:,1)*cfg.tstep;
    
    if(is_individual)
          Det_flux_mat(1,s) = dflux(srcs(s,1)-10,srcs(s,2)-10);
          Det_flux_mat(2,s) = dflux(srcs(s,1)-10,srcs(s,2)+10);
          Det_flux_mat(3,s) = dflux(srcs(s,1)+10,srcs(s,2)+10);
          Det_flux_mat(4,s) = dflux(srcs(s,1)+10,srcs(s,2)-10);
          
          Det_loc_mat{1,s} = [srcs(s,1)-10,srcs(s,2)-10];
          Det_loc_mat{2,s} = [srcs(s,1)-10,srcs(s,2)+10];
          Det_loc_mat{3,s} = [srcs(s,1)+10,srcs(s,2)+10];
          Det_loc_mat{4,s} = [srcs(s,1)+10,srcs(s,2)-10];
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
save('2sparse_DOT','Det_flux');