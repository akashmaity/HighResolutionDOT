function fun_synthesize_dv( D_VEIN , gpuID, unitinmm , N ) 
% scene: the vein is in the center of the image, placed in parallel with the rows 
% The projection lines are slit patterns 

    %clear all;
    close all; 

    %---------Scene description  ----------%
    % N = 100; % volume size is N x N x N
    
    cube_vein = 0;   % if use cube vein
    if_vertical = 0; % if vertical vein
    
    %D_VEIN = [10, 6, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ];% vein depth: distance from the skin surface: depth = d_vein * unitinmm  mm
    r_vein = 3 % 10;  % vein radius 
    cfg.nphoton = 1e8; % 2e8; % # of photons
    % ---------
    
    cfg.gpuid= gpuID ;
    nphoton = cfg.nphoton;
    Lx = linspace(1, N-1, N-1); % row position for the projection line 
    
    z_surf = 20;  % should be integer ! the interface between the air and the skin is at z_surf
    for d_vein = D_VEIN
        z_vein = d_vein + z_surf; 
        % NOTE: The properties are in 1/mm 
        % rows: air (x) ; water ; blood; dermis; epidermis 
                   % mu_a        mu_s       g      refractive index
        prop=[ 0.0000          0.0        1.0000    1
               0.0000          0.0        1.0000    1 
               2.3             10         0.9000    1.3700
               0.0458/4        10         0.9000    1.3700
               1.6572/4        10         0.9000    1.3700 ];
        % mu_a        mu_s       g         refractive index
        %prop=[ 0.0000         0.0        1.0000    1
        %       3.5640e-05     1.0000     1.0000    1.3700
        %       23.0543        9.3985     0.9000    1.3700
        %       0.0458         35.6541    0.9000    1.3700
        %       1.6572         37.5940    0.9000    1.3700]; 
    
        cfg.prop = prop;
        cfg.unitinmm = unitinmm % 0.005; 
        scene_setup.nphoton = cfg.nphoton; scene_setup.N = N; scene_setup.Lx = Lx;
        scene_setup.z_surf = z_surf; scene_setup.d_vein = d_vein;
        scene_setup.r_vein = r_vein; scene_setup.prop = prop; scene_setup.unitinmm = cfg.unitinmm; 
        cfg.vol=zeros(N,N,N);
    
        %-----------------------%
        % Assuming the skin is without layers %
        if cube_vein == 1
          str_vein  = sprintf('{"Box": {"Tag":2, "O":[10, 10, 30], "Size":[5,5,5] }}]}' );
        else
          if if_vertical == 0
            str_vein = sprintf('{"Cylinder": {"Tag":2, "C0": [%f, 0, %d], "C1": [%f, %d, %d], "R": %d}}]}', ...
                                    N/2, z_vein, N/2, N, z_vein, r_vein );
          else
            str_vein = sprintf('{"Cylinder": {"Tag":2, "C0": [0, %f, %d], "C1": [%d, %f, %d], "R": %d}}]}', ...
                                    N/2 + 3, z_vein, N, N/2 - 3, z_vein, r_vein );
          end
        end 
        str_zlayers = sprintf('{"Shapes":[ {"ZLayers":[[1,%d, 1],[%d, %d, 3] ]},', z_surf, z_surf+1, N);

        % scene with vein structure
        cfg.shapes=[str_zlayers  str_vein]; 

        % debug: only the homogenous material 
        %str_zlayers = sprintf('{"Shapes":[ {"ZLayers":[[1,%d, 1],[%d, %d, 3] ]}]}', z_surf, z_surf+1, N);
        %cfg.shapes=[str_zlayers]; 

        cfg.tstep=5e-8;
        cfg.srctype='slit'; % planar
        cfg.srcdir=[0 0 1];
        cfg.srcparam1=[0 N 0 0];
        cfg.isreflect=0;
        cfg.issrcfrom0=1;
        cfg.tstart=0;
        cfg.tend=5e-8;
        cfg.autopilot=1;
        cfg.debuglevel='P';
        
        %cfg.outputtype='energy';
        cfg.outputtype = 'flux'; 
        Imgs = zeros( N, N, length(Lx) );
        i = 0; 
        MCX_DATA = zeros( N, N, N, length( Lx ) );

        for lx = Lx
            i = i+1;
            cfg.srcpos= [lx 0 0];
            fprintf('\n\n\n %d/%d images, lx = %f \n\n\n', i, length( Lx ), lx );
            flux=mcxlab(cfg);

            % convert mcx solution to mcxyz's output
            % 'energy': mcx outputs normalized energy deposition, must convert
            % it to normalized energy density (1/cm^3) as in mcxyz
            % 'flux': cfg.tstep is used in mcx's fluence normalization, must 
            % undo 100 converts 1/mm^2 from mcx output to 1/cm^2 as in mcxyz
    
            if(strcmp(cfg.outputtype,'energy'))
                mcxdata=flux.data/((cfg.unitinmm/10)^3);
            else
                mcxdata=flux.data;
            end
    
            if(strcmp(cfg.outputtype,'flux'))
                mcxdata=mcxdata*cfg.tstep;
            end
    
            Imgs(:, :, i) = mcxdata(:, :, z_surf+ 1);
            img_tmp = Imgs(:, :, i);
            %fprintf('max flux, min flux: %f, %f\n', max( mcxdata(:) ), min(mcxdata(:)) );

            % save the mcx 3D data 
            MCX_DATA(:, :, :, i) = mcxdata;
            fprintf('max img, min img: %f, %f\n', max( img_tmp(:) ), min(img_tmp(:)) );
        end
        
        fprintf('saving the results... \n')
        scene_setup.nphoton = cfg.nphoton; 
        scene_setup.N = N; 
        scene_setup.Lx = Lx; 
        scene_setup.z_surf = z_surf; 
        scene_setup.d_vein = d_vein;
        scene_setup.r_vein = r_vein;
        scene_setup.prop = prop; 
        scene_setup.unitinmm = cfg.unitinmm;
        unitinmm = cfg.unitinmm;
        
        if cube_vein == 1
            fname = sprintf( 'dat/mcx_imgs_cube_vein_N%d_vd_%.2f_vr_%.2f.mat', scene_setup.N, d_vein, r_vein );
        elseif if_vertical==0
            fname = sprintf( 'dat/mcx_imgs_N%d_vd_%.2f_vr_%.2f.mat', scene_setup.N, d_vein, r_vein );
        else
            fname = sprintf( 'dat/mcx_imgs_N%d_vd_%.2f_vr_%.2f_vertical.mat', scene_setup.N, d_vein, r_vein );
        end 

        fprintf('result path: %s', fname);
        save(fname, 'Imgs', 'N', 'Lx', 'prop', 'unitinmm', 'z_surf', 'd_vein', 'r_vein', 'nphoton', 'z_surf', 'MCX_DATA');
        fprintf('Done \n') 
    end 

end
