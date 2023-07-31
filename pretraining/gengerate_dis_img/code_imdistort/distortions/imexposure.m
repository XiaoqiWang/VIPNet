function J = imexposure(I, expo, doLab)
    
if exist('doLab','var') && doLab
    lab = rgb2lab(I);
    L = lab(:,:,1)/100;

    L_ = truncate(L .* (2^expo));

    lab(:,:,1) = L_*100;
    J = truncate(lab2rgb(lab));
else
    J = truncate( mapmm(I) .* (2^expo) );
end

