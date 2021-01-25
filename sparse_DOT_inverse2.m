clear all; 
tic
load('2sparse_DOT.mat')
is_individual = 1;
g = 0.9;
mf = 0.25;
x = 128*mf;
y = 128*mf;
z = 32*mf;
scale = 1/3;
mua = 0.01145;
musp = 10*(1-g);
Reff = 0;
D = 1/(3*(mua+musp));
beta = sqrt(3*musp*mua);



z0 = 1/(musp+mua);
zb = (1+Reff)/(1-Reff)*2*D;
 

src_r = 2:2:10;
src_c = 2:2:10;
[sr,sc] = ndgrid(src_r,src_c);
srcs = [sr(:),sc(:)]*10*scale;

det_r = 1:2:11;
det_c = 1:2:11;
[dr,dc] = ndgrid(det_r,det_c);
dets = [dr(:),dc(:)]*10*scale;


no_dts = length(dets);
no_srcs = length(srcs);
%% get Kernel for full image size
if(~is_individual)
for i = 1: no_dts
    i
    %A = zeros(x,y,z);
    for mr = 1:x
        m = mr / mf * scale;
        for nr = 1:y
            n = nr / mf * scale;
            for or = 1:z
                o = or / mf * scale;
                for src_idx = 1:no_srcs                        
                    RS1 = sqrt((m-srcs(src_idx,1))^2+(n-srcs(src_idx,2))^2+(o-z0)^2);
                    RS2 = sqrt((m-srcs(src_idx,1))^2+(n-srcs(src_idx,2))^2+(o+z0+2*zb)^2);
                    
                    RD1 = sqrt((m-dets(i,1))^2+(n-dets(i,2))^2+(0-o)^2);
                    RD2 = sqrt((m-dets(i,1))^2+(n-dets(i,2))^2+(0-2*zb-o)^2);
                    
                    RSD1 = sqrt((dets(i,1)-srcs(src_idx,1))^2+(dets(i,2)-srcs(src_idx,2))^2+(z0)^2);
                    RSD2 = sqrt((dets(i,1)-srcs(src_idx,1))^2+(dets(i,2)-srcs(src_idx,2))^2+(-z0-2*zb)^2);
                    
                    Phi1 = 1/(4*pi*D)*(exp(-beta*RS1)/RS1-exp(-beta*RS2)/RS2);
                    Phi2 = 1/(4*pi*D)*(exp(-beta*RD1)/RD1-exp(-beta*RD2)/RD2);
                    Phi0 = 1/(4*pi*D)*(exp(-beta*RSD1)/RSD1-exp(-beta*RSD2)/RSD2);
                    
                    G(mr,nr,or,src_idx) = Phi1.*Phi2./Phi0;
                end
            end
        end
    end
    A{i} = sum(G,4);
end
else
    for mr = 1:x
        mr
        m = mr / mf * scale;
        for nr = 1:y
            n = nr / mf * scale;
            for or = 1:z
                o = or / mf * scale;
                for src_idx = 1:no_srcs
                    dets(1,1) = srcs(src_idx,1) - 10*scale;
                    dets(1,2) = srcs(src_idx,2) - 10*scale;
                    dets(2,1) = srcs(src_idx,1) - 10*scale;
                    dets(2,2) = srcs(src_idx,2) + 10*scale;
                    dets(3,1) = srcs(src_idx,1) + 10*scale;
                    dets(3,2) = srcs(src_idx,2) + 10*scale;
                    dets(4,1) = srcs(src_idx,1) + 10*scale;
                    dets(4,2) = srcs(src_idx,2) - 10*scale;
                    for i = 1:4
                        
                    RS1 = sqrt((m-srcs(src_idx,1))^2+(n-srcs(src_idx,2))^2+(o-z0)^2);
                    RS2 = sqrt((m-srcs(src_idx,1))^2+(n-srcs(src_idx,2))^2+(o+z0+2*zb)^2);
                    
                    RD1 = sqrt((m-dets(i,1))^2+(n-dets(i,2))^2+(0-o)^2);
                    RD2 = sqrt((m-dets(i,1))^2+(n-dets(i,2))^2+(0-2*zb-o)^2);
                    
                    RSD1 = sqrt((dets(i,1)-srcs(src_idx,1))^2+(dets(i,2)-srcs(src_idx,2))^2+(z0)^2);
                    RSD2 = sqrt((dets(i,1)-srcs(src_idx,1))^2+(dets(i,2)-srcs(src_idx,2))^2+(-z0-2*zb)^2);
                    
                    Phi1 = 1/(4*pi*D)*(exp(-beta*RS1)/RS1-exp(-beta*RS2)/RS2);
                    Phi2 = 1/(4*pi*D)*(exp(-beta*RD1)/RD1-exp(-beta*RD2)/RD2);
                    Phi0 = 1/(4*pi*D)*(exp(-beta*RSD1)/RSD1-exp(-beta*RSD2)/RSD2);
                    
                    G(mr,nr,or,src_idx,i) = Phi1.*Phi2./Phi0;
                    end
                end
            end
        end
    end
    %A{i} = sum(G,4);
%     A{i} = G;
end


%% get homogeneous getIntensity
if(~is_individual)
    for i = 1: no_dts
        i
        for src_idx = 1:no_srcs
            
            RSD1 = sqrt((dets(i,1)-srcs(src_idx,1))^2+(dets(i,2)-srcs(src_idx,2))^2+(z0-0.5)^2);
            RSD2 = sqrt((dets(i,1)-srcs(src_idx,1))^2+(dets(i,2)-srcs(src_idx,2))^2+(-z0-2*zb-0.5)^2);
            
            Phi0 = 1/(4*pi*D)*(exp(-beta*RSD1)/RSD1-exp(-beta*RSD2)/RSD2);
                       
            
            fx(i,src_idx) = Phi0;
        end
        
    end
    I0 = sum(fx,2);
else
    RSD1 = sqrt((10*scale)^2+(10*scale)^2+(z0-0.5)^2);
    RSD2 = sqrt((10*scale)^2+(10*scale)^2+(-z0-2*zb-0.5)^2);
    
    Phi0 = 1/(4*pi*D)*(exp(-beta*RSD1)/RSD1-exp(-beta*RSD2)/RSD2);
    I0 = Phi0*ones(size(Det_flux));
end
%% optimize for Q
if(is_individual)
    count = 0;
    for i = 1:no_srcs
        for j = 1:4
            matj = G(:,:,:,:,j);
            mat_src_i= matj(:,:,:,i);
            W(j+count,:) = mat_src_i(:);
        end
        count = count + 4;
    end
else
    count = 0;
    for j = 1:no_dts
        matj = A{j};
        W(j+count,:) = matj(:);
    end
    count = count + no_dts;
end
    
    

lamda  = 2;
Q_hat = inv(W'*W+lamda * eye(length(W)))*W'*(1-Det_flux./I0);
%Q_hat = ridge(1-Det_flux./I0,W,lamda)
Q = reshape(Q_hat,x,y,z);

imagesc(Q(:,:,4),[0 0.017]);
colormap('jet');
save('2inverseDOT2.mat','Q');
toc