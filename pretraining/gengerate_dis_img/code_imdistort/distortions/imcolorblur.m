function J = imcolorblur(I, sigma, scaling)

if ~exist('scaling','var')
    scaling = 1;
end

lab = rgb2lab(I);

lab(:,:,2:3) = imgaussfilt(lab(:,:,2:3), sigma) .* scaling;

J = truncate(lab2rgb(lab));

