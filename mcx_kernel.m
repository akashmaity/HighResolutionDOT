load('Kernel.mat');
W_r = W(35:66,35:66,1:8,:);

x0 = [1:32;1:32;1:8];
xi = [1:32;1:32;1:0.25:8];

for i = 1:81
    kernel(:,:,:,i) = imresize3(W_r(:,:,:,i),[32,32,32]);
end
save('Kernel_modified.mat')