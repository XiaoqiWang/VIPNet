function b =  imcut(a, border)
% b =  imcut(a, border)

if any(border)
    bh = floor(border/2);
    b = a(bh+1:end-bh, bh+1:end-bh, :, :);
else
    b = a;
end