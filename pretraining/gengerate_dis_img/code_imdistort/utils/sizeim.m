function [w, h] = sizeim(im,flip) %#codegen 

if nargin<2, flip = false; end

if isa(im,'dip_image')
    sz = [size(im,2) size(im,1)];
elseif isa(im, 'dip_image_array') || iscell(im)
    sz = sizeim(im{1});
else
    sz = [size(im,1) size(im,2)];
end

if flip, sz = fliplr(sz); end

if nargout>1
    [w, h] = dealarray(sz);
else
    w = sz;
end