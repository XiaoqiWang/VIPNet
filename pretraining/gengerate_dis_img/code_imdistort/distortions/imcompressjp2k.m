function J = imcompressjp2k(I, ratio)


fname = tempname;
imwrite(I,fname,'jp2','CompressionRatio', ratio)
J = im2double(imread(fname));
delete(fname);


end