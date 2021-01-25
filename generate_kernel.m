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
diffuse = 0;
cfg.unitinmm  = 1;
cfg.isreflect = 1;
cfg.isnormalized = 1;

N = 100;
Nr = 100;
Nc = 100;
Nz = 100;


%src_row = 20;

% cfg.vol=zeros(Nr,Nc,Nz);
cfg.vol=uint8(ones(Nr,Nc,Nz));
if(diffuse)
    cfg.vol(:,:,1)=0;
end

cfg.prop=[0.0000         0.0    1.0000    1
    0.01   10    0.9000    1.3700
    0.0000   0.0   1.0000    1];

% cfg.prop=[0.0000         0.0    1.0000    1
%         0.01  4    0.9000    1.3700
%         0.000   4    0.9000    1.3700];
    
cfg.tstart=0;
cfg.tend=1e-7;
cfg.tstep=1e-7;
cfg.srcdir=[0 0 1];

cfg.autopilot=1;
cfg.gpuid=1;
cfg.debuglevel='P';



%cfg.outputtype='energy';
cfg.outputtype='flux';
cfg.nphoton=1e8;
cfg.issrcfrom0=0;
cfg.issaveref = 0; % save diffuse reflectance


cfg.srcpos=[N/2 N/2 0];
cfg.srctype='pencil';
flux=mcxlab(cfg);

Phi_2 = flux.data*cfg.tstep;

Lx = 1:N/2;
for lx = Lx
    %% Source is a slit
    if(src_slit==1)
        cfg.srcpos=[lx 34 0];
        cfg.srctype='slit';
        cfg.srcparam1=[0 33 0 0];
    else
        %% Source is a pencil beam
        cfg.srcpos=[src_row Nc/2 0];
        cfg.srctype='pencil';
        cfg.srcparam1=[0 0 0 0];
    end
    
    
    flux=mcxlab(cfg);

    
    if(strcmp(cfg.outputtype,'energy'))
        mcxdata=flux.data/((cfg.unitinmm/10)^3);
    else
        mcxdata=flux.data; % equivalent to placing 100 sources in diffusive model
        mcxdiffuse=flux.dref;
    end
    
    if(strcmp(cfg.outputtype,'flux'))
        if(diffuse)
            mcxdiffuse=mcxdiffuse*cfg.tstep;
        else
            mcxdata=mcxdata*cfg.tstep;
        end
    end
    Phi_1 = mcxdata;
    
    Kernel(lx,:,:,:) = Phi_1.*Phi_2;
end

save('MCX_Kernel_650.mat');
load('MCX_Kernel_650.mat');
load('kernel_dipole.mat');

clim = [5e-9 1e-4];
imagesc(squeeze(Kernel_model(10,:,16,:))',clim)
figure
imagesc(squeeze(Kernel(40,:,50,:))',clim)

K1 = squeeze(Kernel_model(10,:,:,:));
K2 = squeeze(Kernel(40,:,:,:));

K2m = K2(50-16:50+16,50-16:50+16,1:33);

imagesc(squeeze(K1(:,16,:))')
figure
imagesc(squeeze(K2m(:,16,:))')

clim = [0 1];
V = abs(squeeze(K1(:,16,:))-squeeze(K2m(:,16,:)))./abs(squeeze(K2m(:,16,:)));
imagesc(V',clim);

colorbar


load('MCX_Kernel_650.mat');
clim = [5e-9 9e-5];
imagesc(squeeze(Kernel(40,:,50,:))',clim)
colormap(jet);
figure
load('MCX_Kernel_950.mat');
imagesc(squeeze(Kernel(40,:,50,:))',clim)
colormap(jet);

