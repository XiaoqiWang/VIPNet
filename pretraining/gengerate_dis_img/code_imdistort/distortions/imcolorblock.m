function [b] = imcolorblock(a,pnum)

patch_size = [32 32];

[h,w,~] = size(a);

b = a;

h_max = h - patch_size(1);
w_max = w - patch_size(2);

for i = 1:pnum
    color = rand(1,3);
    x = rand * w_max;
    y = rand * h_max;
    b = insertShape(b,'FilledRectangle',[x y patch_size],'Color',color,'Opacity',1.0);
end


end