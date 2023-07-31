function J = imbrighten(I, amount, dolab)
% amount in [0, 1]

I = im2double(I);
dolab = nchans(I)==3 || (exist('dolab','var') && dolab);

if dolab
    lab = rgb2lab(I);
    L = lab(:,:,1)/100;
    L_ = curves(L, 0.5 + amount/2);
    lab(:,:,1) = L_*100;
end

J = curves(I, 0.5 + amount/2);

if dolab
    J = (2*J + truncate(lab2rgb(lab)))/3;
end
