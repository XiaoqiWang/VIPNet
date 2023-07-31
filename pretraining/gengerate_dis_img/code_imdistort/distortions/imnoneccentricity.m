function [b] = imnoneccentricity(a,pnum)


patch_size = [16 16];
radius = 16;

[h,w,~] = size(a);
 

b = a;

h_min = radius;
w_min = radius;

h_max = h - patch_size(1) - radius;
w_max = w - patch_size(2) - radius;

for i = 1:pnum
    w_start = round(rand * (w_max - w_min)) + w_min;
    h_start = round(rand * (h_max - h_min)) + h_min;
    patch = b(h_start:h_start+patch_size(1)-1,w_start:w_start+patch_size(1)-1,:);
    
    rand_w_start = round((rand - 0.5)*radius + w_start);
    rand_h_start = round((rand - 0.5)*radius + h_start);
    b(rand_h_start:rand_h_start+patch_size(1)-1,rand_w_start:rand_w_start+patch_size(1)-1,:) = patch;
end



end