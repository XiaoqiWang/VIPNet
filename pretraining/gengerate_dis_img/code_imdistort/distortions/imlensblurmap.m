function z = imlensblurmap(x, map, MaxBlur)

if ~exist('MaxBlur','var')
    MaxBlur = 10;
end
rads = linspace(1, MaxBlur, 10);
y = mapimx(@(rad)imblur(x, 'lens', rad), rads);
y = cat(3, x, y);

dm = 1-map;
dm = double(erosion(dilation(dm, 3), 20));
dm = mapmm(double(imgaussian(dm,10)));
dm = floor(dm*10) + 1;
[xx, yy] = ndgrid(1:nrows(dm), 1:ncols(dm));
z = reshape(sref(y, [xx(:) yy(:) dm(:)]), sizeim(dm));
