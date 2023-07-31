function b = imblurlens(a, radius)

k = fspecial('disk', radius);
b = imfilter(a, k, 'replicate','same');