function J = imcompressjpeg(I, quality)


fname = tempname;
imwrite(I,fname,'jpg','quality', quality)
J = im2double(imread(fname));
delete(fname);


end