function z = imwarpmap(im, shiftmap, interpType, padme)
%IMWARPMAP Warp image according to shiftmap.
% 
% Z = imwarpmap(IM, SHIFTMAP)
%
% IM: single channel image
% SHIFTMAP: 2-channel image describing the local shifts of each pixel along X and Y directions

sy = shiftmap(:,:,1); sx = shiftmap(:,:,2);
if ~exist('interpType', 'var') || isempty(interpType)
    interpType = 'bilinear';
end
    
if exist('padme', 'var') && padme        
    margins = ceil([max2(abs(sx)) max2(abs(sy))]);
    im = padarray(im, margins, 'replicate', 'both');
    sy = padarray(sy, margins, 'replicate', 'both');
    sx = padarray(sx, margins, 'replicate', 'both');
end

[xx, yy] = meshgrid(1:size(im,2),1:size(im,1));
z = im*0;
for i=1:nchans(im)
    z(:,:,i) = interp2(im(:,:,i),xx-sx,yy-sy,interpType);
end

if exist('padme', 'var') && padme        
    z = unpadarray(z, margins, 'both');
end