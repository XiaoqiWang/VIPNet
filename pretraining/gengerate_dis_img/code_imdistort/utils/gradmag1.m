function y = gradmag1(im)

y = sqrt(sum(gradientxy(im).^2,3));