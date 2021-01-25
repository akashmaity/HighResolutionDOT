clims = [0 15];
[valmap,dmap] = max(Q_opt,[],1);
dmap(squeeze(valmap)<0.15)=0;
imagesc(squeeze(dmap));

figure
plot((1:33)*0.24,squeeze(Q_opt(:,168,128)),'LineWidth',1.5);
hold on
plot((1:33)*0.24,squeeze(Q_opt(:,87,128)),'LineWidth',1.5);
hold on
plot((1:33)*0.24,squeeze(Q_opt(:,128,128)),'LineWidth',1.5);
set(gca,'Color','k');

% for d = 1:32
%     dmap(d,:,:) = d*Q_opt(d,:,:);
% end
% 
% clims = [0 15];
% depth_map = squeeze((sum(dmap,1)./sum(Q_opt,1)));
% [valmap,~] = max(Q_opt,[],1);
% depth_map(squeeze(valmap)<0.00092)=0;
% %depth_map = depth_map/18*15;
% imagesc(depth_map*0.24);