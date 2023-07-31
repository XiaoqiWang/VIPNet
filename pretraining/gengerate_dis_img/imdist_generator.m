function [distort_I] = imdist_generator(im, dist_type, dist_level)
% given the image, distortion type id and distortion level, generate
% distorted image

im = mapmm(im);
switch dist_type
    case 1
        levels = [0.1, 0.5, 1, 2, 5];
        distort_I = imblurgauss(im, levels(dist_level));
    case 2
        levels = [1, 2, 4, 6, 8];
        distort_I = imblurlens(im, levels(dist_level));
    case 3
        levels = [1, 2, 4, 6, 10];
        distort_I = imblurmotion(im, levels(dist_level));
    case 4
        levels = [1, 3, 6, 8, 12];
        distort_I = imcolordiffuse(im, levels(dist_level));
    case 5
        levels = [1, 3, 6, 8, 12];
        distort_I = imcolorshift(im, levels(dist_level));
    case 6
        levels = [64, 48, 32, 16, 8];
        distort_I = imcolorquantize(im, levels(dist_level));
    case 7
        levels = [0.4, 0.2, 0.1, 0, -0.4];
        distort_I = imcolorsaturate(im, levels(dist_level));
    case 8
        levels = [1, 2, 3, 6, 9];
        distort_I = imsaturate(im, levels(dist_level));
    case 9
        levels = [16, 32, 45, 120, 400];
        distort_I = imcompressjp2k(im, levels(dist_level));
    case 10
        levels = [43, 36, 24, 7, 4];
        distort_I = imcompressjpeg(im, levels(dist_level));
    case 11
        levels = [0.001, 0.002, 0.003, 0.005, 0.01];
        distort_I = imnoisegauss(im, levels(dist_level));
    case 12
        levels = [0.0001, 0.0005, 0.001, 0.002, 0.003];
        distort_I = imnoisecolorcomp(im, levels(dist_level));
    case 13
        levels = [0.001, 0.005, 0.01, 0.02, 0.03];
        distort_I = imnoiseimpulse(im, levels(dist_level));
    case 14
        levels = [0.001, 0.005, 0.01, 0.02, 0.05];
        distort_I = imnoisemultiplicative(im, levels(dist_level));
    case 15
        levels = [0.01, 0.03, 0.05, 0.1, 0.15];
        distort_I = imdenoise(im, levels(dist_level));
    case 16
        levels = [0.1, 0.2, 0.4, 0.7, 1.1];
        distort_I = imbrighten(im, levels(dist_level));
    case 17
        levels = [0.05, 0.1, 0.2, 0.4, 0.8];
        distort_I = imdarken(im, levels(dist_level));
    case 18
        levels = [0.15, 0.08, 0, -0.08, -0.15];
        distort_I = immeanshift(im, levels(dist_level));
    case 19
        levels = [0.05, 0.1, 0.2, 0.5, 1];
        distort_I = imjitter(im, levels(dist_level));
    case 20
        levels = [20, 40, 60, 80, 100];
        distort_I = imnoneccentricity(im, levels(dist_level));
    case 21
        levels = [0.01, 0.05, 0.1, 0.2, 0.5];
        distort_I = impixelate(im, levels(dist_level));
    case 22
        levels = [20, 16, 13, 10, 7];
        distort_I = imnoisequantize(im, levels(dist_level));
    case 23
        levels = [2, 4, 6, 8, 10];
        distort_I = imcolorblock(im, levels(dist_level));
    case 24
        levels = [1, 2, 3, 6, 12];
        distort_I = imsharpenHi(im, levels(dist_level));
    case 25
        levels = [0.3, 0.15 0, -0.4, -0.6];
        distort_I = imcontrastc(im, levels(dist_level));
    otherwise
        error('Unknown distortion type!')
end
distort_I = mapmm(distort_I);

end