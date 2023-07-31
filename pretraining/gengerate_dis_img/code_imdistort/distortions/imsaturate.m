function J = imsaturate(I, S)

lab = rgb2lab(I);
lab(:,:,2:3) = lab(:,:,2:3) * S;
J = truncate(lab2rgb(lab));

