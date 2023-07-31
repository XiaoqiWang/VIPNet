function J = imreframe(I, aspectFrame)

[h, w, ~]  = size(I);
aspect = w/h;

if aspect > aspectFrame
    szCrop = h*[1 aspectFrame];
else
    szCrop = w*[1/aspectFrame 1];
end

J = imcrop1(I, [h w]/2, szCrop);