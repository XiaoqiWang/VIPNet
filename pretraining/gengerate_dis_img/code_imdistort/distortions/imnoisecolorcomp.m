function J = imnoisecolorcomp(I, var)

YCBCR = rgb2ycbcr(I);
J = ycbcr2rgb(imnoise(YCBCR,'gaussian', 0, var));

end