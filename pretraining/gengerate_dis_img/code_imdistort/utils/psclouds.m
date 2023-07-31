function r = psclouds(sz, sigma)
%PSCLOUDS PhotoShop style clouds effect
%
% r = psclouds(sz, sigma)
%
% Generates a multi-channel low frequency (blurred) std normal noise image

r = mat2im((randn(sz)-0.5)*2);
if size(r,3)>1
    for j=0:size(r,3)-1
            r(:,:,j) = gaussf(r(:,:,j), sigma);
    end
else
    r = gaussf(r, sigma);
end

r = double(mapmm(r,-1,1));
