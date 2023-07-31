function [ distorted_img ] = distortion_generator( img, dist_type, level )
    %% set distortion parameter
    gblur_level = [7,15,39,91,199];
    wn_level = [-10,-7.5,-5.5,-3.5,0];
    jpeg_level = [43,12,7,4,0];
    jp2k_level = [0.46,0.16,0.07,0.04,0.02]; % bit per pixel
    
    %% distortion generation
    switch dist_type
        case 1
            hsize = gblur_level(level);
            h = fspecial('gaussian', hsize, hsize/6);
            distorted_img = imfilter(img,h,'symmetric');
        case 2
            distorted_img = imnoise(img,'gaussian',0,2^(wn_level(level)));
        case 3
            testName = [num2str(randi(intmax)) '.jpg'];
            imwrite(img,testName,'jpg','quality',jpeg_level(level));
            distorted_img = imread(testName);
            delete(testName);
        case 4
            testName = [num2str(randi(intmax)) '.jp2'];
            imwrite(img,testName,'jp2','CompressionRatio', 24 / jp2k_level(level));
            distorted_img = imread(testName);
            delete(testName);
    end
end

