function [b] = imcolorsaturate(a,factor)

hsvIm = rgb2hsv(a);
hsvIm(:,:,2) = hsvIm(:,:,2) * factor;
b = hsv2rgb(hsvIm);

end