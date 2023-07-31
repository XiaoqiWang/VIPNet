function J = imdenoise(I,var)

net = denoisingNetwork('DnCNN');



noisyRGB = imnoise(I,'gaussian',0,var);

noisyR = noisyRGB(:,:,1);
noisyG = noisyRGB(:,:,2);
noisyB = noisyRGB(:,:,3);

denoisedR = denoiseImage(noisyR,net);
denoisedG = denoiseImage(noisyG,net);
denoisedB = denoiseImage(noisyB,net);

J = cat(3,denoisedR,denoisedG,denoisedB);


end