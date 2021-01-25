%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  mcxyz skinvessel benchmark
%
%  must change mcxyz maketissue.m boundaryflag variable from 2 to 1 to get
%  comparable absorption fraction (40%), otherwise, mcxyz obtains slightly
%  higher absorption (~42%) with boundaryflag=2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
addpath(genpath('/home/akash/mcx/mcx_bin') )
%load mcxyz_skinvessel.mat

Nr = 100;
Nc = 100;
Nz = 33;
z_vein = 3;
r_vein = 1;
if_vertical = 0;
src_slit = 1;
src_row = 20;

cfg.vol=zeros(Nr,Nc,Nz);

str_zlayers = sprintf('{"Shapes":[ {"ZLayers":[[0,%d, 2]]},', Nz);
if if_vertical == 0
  str_vein    = sprintf('{"Cylinder": {"Tag":1, "C0": [%f, 0, %d], "C1": [%f, %d, %d], "R": %d}}]}', ...
                              Nr/2, z_vein, Nr/2, Nc, z_vein, r_vein );
else
  str_vein    = sprintf('{"Cylinder": {"Tag":1, "C0": [0, %f, %d], "C1": [%d, %f, %d], "R": %d}}]}', ...
                              Nc/2 , z_vein, Nr, Nc/2, z_vein, r_vein );

end 

l  = 6;
str_box = sprintf('{"Box":{"Tag":1, "O":[%d,%d,3],"Size":[%d,%d,%d]}}]}',50-3,0,6,100,6);

cfg.shapes=[str_zlayers str_vein] ; 
cfg.unitinmm = 1;

% cfg.prop=[0.00   0     1.0000    1
%           0.1   10    0.9000    1.3700
%           0.01   10    0.9000    1.3700
%           0.01   10    0.9000    1.3700];

cfg.prop=[0.0000         0.0    1.0000    1
   2.03   35.6541    0.9000    1.3700
    0.0458   35.6541    0.9000    1.3700
    1.6572   37.5940    0.9000    1.3700];

cfg.nphoton=1e7;
cfg.issrcfrom0=0;
%% Source is a slit
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

cfg.tstart=0;
cfg.tend=1e-7;
cfg.tstep=1e-7;
cfg.srcdir=[0 0 1];
cfg.isreflect=1;
cfg.autopilot=1;
cfg.gpuid=1;
cfg.debuglevel='P';



%cfg.outputtype='energy';
cfg.outputtype='flux';
flux=mcxlab(cfg);

% convert mcx solution to mcxyz's output
% 'energy': mcx outputs normalized energy deposition, must convert
% it to normalized energy density (1/cm^3) as in mcxyz
% 'flux': cfg.tstep is used in mcx's fluence normalization, must 
% undo 100 converts 1/mm^2 from mcx output to 1/cm^2 as in mcxyz
if(strcmp(cfg.outputtype,'energy'))
    mcxdata=flux.data/((cfg.unitinmm/10)^3);
else
    mcxdata=flux.data; % equivalent to placing 100 sources in diffusive model
end

if(strcmp(cfg.outputtype,'flux'))
    mcxdata=mcxdata*cfg.tstep;
end


% imagesc(log10(abs(squeeze(mcxdata(:,:,1))))')

% axis equal;
% colormap(jet);
% colorbar
% if(strcmp(cfg.outputtype,'energy'))
%     set(gca,'clim',[-2.4429 4.7581])
% else
%     set(gca,'clim',[0.5 2.8])
% end

for d = 21:80
    det_row = d;
    Intensity(d-20) = log10(abs(squeeze(mcxdata(det_row,Nc/2,1))));
%     [Phi r]= cwdiffusion(0.0458/4, 1, 0, cfg.srcpos*cfg.unitinmm ,[det_row 50 0.5]*cfg.unitinmm);
%     Intensity_cw(d-20) = log10(Phi);
end

plot(Intensity,'*','LineStyle','none','LineWidth',1.5)
hold on
if(src_slit==1)
    load('intensity_dipole_slit.mat');
else
    load('intensity_dipole_point.mat');
end
plot(Id,'LineWidth',1.5)
% hold on
% plot(Intensity_cw,'LineWidth',1.5)
% % 
% % legend('Monte-Carlo','Dipole diffusion','MCX')
% 
% error = norm(Id - Intensity)/norm(Intensity)