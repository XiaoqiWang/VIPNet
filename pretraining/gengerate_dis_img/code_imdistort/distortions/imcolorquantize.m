function J = imcolorquantize(I,n)

[X,map] = rgb2ind(I,n);

J = ind2rgb(X,map);

end