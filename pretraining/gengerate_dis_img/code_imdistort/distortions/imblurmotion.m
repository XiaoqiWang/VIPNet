function b = imblurmotion(a, radius, angle)

if ~exist('angle','var')
    angle = randi([0, 180], 1);
end

k = fspecial('motion', radius, angle);
b = imfilter(a, k, 'replicate','same');
