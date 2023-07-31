function J = imcompress(I, strength, format)
% strength in [0, 1) (= 1 - Q/100)

if ~exist('format','var')
    format = 'JPEG';
end

fname = tempname;
imwrite(mapmm(I), fname, format, 'Quality', (1-strength^0.1)*100);
J = im2double(imread(fname));
delete(fname);

J = imsharpenHi(J, 0.5, 1);