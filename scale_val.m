%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  mcxyz skinvessel benchmark
%
%  must change mcxyz maketissue.m boundaryflag variable from 2 to 1 to get
%  comparable absorption fraction (40%), otherwise, mcxyz obtains slightly
%  higher absorption (~42%) with boundaryflag=2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
addpath(genpath('/home/akm8/Diffuse_Model/mcxlab') )
%load mcxyz_skinvessel.mat
src_slit = 1;
diffuse = 1;
cfg.unitinmm  = 1;
cfg.isreflect = 1;
cfg.isnormalized = 1;

N = 100;
Nr = 100;
Nc = 100;
Nz = 33;


src_row = 20;

% cfg.vol=zeros(Nr,Nc,Nz);
cfg.vol=uint8(ones(Nr,Nc,Nz));
if(diffuse)
    cfg.vol(:,:,1)=0;
end


cfg.prop=[0.0000         0.0    1.0000    1
    0.01   10    0.9000    1.00
    0.0000   0.0   1.0000    1];

cfg.nphoton=1e7;
cfg.issrcfrom0=0;
cfg.issaveref = 1; % save diffuse reflectance

%% Source is a slit
if(src_slit==1)
    cfg.srcpos=[src_row 0 0];
    cfg.srctype='slit';
    cfg.srcparam1=[0 N 0 0];
else
%% Source is a pencil beam
    cfg.srcpos=[src_row Nc/2 0];
    cfg.srctype='pencil';
    cfg.srcparam1=[0 0 0 0];
end

cfg.tstart=0;
cfg.tend=1e-7;
cfg.tstep=1e-7;
cfg.srcdir=[0 0 1];

cfg.autopilot=1;
cfg.gpuid=1;
cfg.debuglevel='P';



%cfg.outputtype='energy';
cfg.outputtype='flux';
flux=mcxlab(cfg);

if(src_slit==1)
    cfg.srcpos=[src_row 0 0];
    cfg.srctype='slit';
    cfg.srcparam1=[0 Nc 0 0];
else
%% Source is a pencil beam
    cfg.srcpos=[src_row Nc/2 0];
    cfg.srctype='pencil';
    cfg.srcparam1=[0 0 0 0];
end

% convert mcx solution to mcxyz's output
% 'energy': mcx outputs normalized energy deposition, must convert
% it to normalized energy density (1/cm^3) as in mcxyz
% 'flux': cfg.tstep is used in mcx's fluence normalization, must 
% undo 100 converts 1/mm^2 from mcx output to 1/cm^2 as in mcxyz

mcxdata=flux.data; % equivalent to placing 100 sources in diffusive model
mcxdiffuse=flux.dref;

if(strcmp(cfg.outputtype,'flux'))
    if(diffuse)
        mcxdiffuse=mcxdiffuse*cfg.tstep;
    else
        mcxdata=mcxdata*cfg.tstep;
    end
end


for d = 21:80
    det_row = d;
    if(diffuse)
        Intensity(d-20) = log10(abs(squeeze(mcxdiffuse(det_row,Nc/2,1))));
    else 
        Intensity(d-20) = log10(abs(squeeze(mcxdata(det_row,Nc/2,1))));
    end
%     [Phi r]= cwdiffusion(0.0458/4, 1, 0, cfg.srcpos*cfg.unitinmm ,[det_row 50 0.5]*cfg.unitinmm);
%     Intensity_cw(d-20) = log10(Phi);
end
% 


plot((1:60)*cfg.unitinmm,Intensity,'*','LineStyle','none','LineWidth',1.5)
hold on
% if(src_slit==1)
%     load(sprintf('intensity_dipole_slit_mf_%.1f.mat',cfg.unitinmm));
% else
%     load(sprintf('intensity_dipole_mf_%.1f.mat',cfg.unitinmm));
% end
% plot((1:60)*cfg.unitinmm,Id,'LineWidth',1.5);
% 
% hold on


cfg.unitinmm  = 0.24;
N = 300;
Nr = 300;
Nc = 300;
Nz = 33;



% cfg.vol=zeros(Nr,Nc,Nz);
cfg.vol=uint8(ones(Nr,Nc,Nz));
if(diffuse)
    cfg.vol(:,:,1)=0;
end

if(src_slit==1)
    cfg.srcpos=[src_row 0 0];
    cfg.srctype='slit';
    cfg.srcparam1=[0 Nc 0 0];
end

    
flux=mcxlab(cfg);


mcxdata=flux.data; % equivalent to placing 100 sources in diffusive model
mcxdiffuse=flux.dref;

if(strcmp(cfg.outputtype,'flux'))
    if(diffuse)
        mcxdiffuse=mcxdiffuse*cfg.tstep;
    else
        mcxdata=mcxdata*cfg.tstep;
    end
end


for d = 21:80
    det_row = d;
    if(diffuse)
        Intensity(d-20) = log10(abs(squeeze(mcxdiffuse(det_row,Nc/2,1))));
    else 
        Intensity(d-20) = log10(abs(squeeze(mcxdata(det_row,Nc/2,1))));
    end
end


plot((1:60)*cfg.unitinmm,Intensity,'*','LineStyle','none','LineWidth',1.5)
% hold on
% if(src_slit==1)
%     load(sprintf('intensity_dipole_slit_mf_%.1f.mat',cfg.unitinmm));
% else
%     load(sprintf('intensity_dipole_mf_%.1f.mat',cfg.unitinmm));
% end
% plot((1:60)*cfg.unitinmm,Id,'LineWidth',1.5);
