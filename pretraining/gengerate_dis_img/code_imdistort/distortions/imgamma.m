function J = imgamma(I, gamma) 

lab = rgb2lab(I);
L = lab(:,:,1)/100;

L_ = truncate((L+1).^gamma - 1);

lab(:,:,1) = L_*100;
J = truncate(lab2rgb(lab));