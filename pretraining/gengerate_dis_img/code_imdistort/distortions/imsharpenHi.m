function J = imsharpenHi(I, amount, radius)

if ~exist('radius','var')
    radius = 3;
end

J = imsharpen(I, 'Radius', radius, 'Amount', amount);
J = truncate(J);