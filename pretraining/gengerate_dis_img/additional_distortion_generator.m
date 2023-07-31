function distorted_img = additional_distortion_generator(img, dist_type, level)
    %% set distortion parameter
    pink_level = [0.061,0.127,0.211,0.316,0.543];
%   ca_level = [1,2,4,5,9];
    oe_level = [2,6,9,12,17];
    ue_level = [2,6,9,12,17];
    %% distortion generation
    switch dist_type
        case 26 % pink noise
          h = size(img,1);
            w = size(img,2);    
            
            fnoise_R = randnd(-1,2^9);fnoise_G = randnd(-1,2^9);fnoise_B = randnd(-1,2^9);
            fnoise_R = fnoise_R/max(abs(fnoise_R(:)));
            fnoise_G = fnoise_G/max(abs(fnoise_G(:)));
            fnoise_B = fnoise_B/max(abs(fnoise_B(:)));
            fnoise2 = zeros(512,512,3);
            fnoise2(:,:,1) = fnoise_R;fnoise2(:,:,2) = fnoise_G;fnoise2(:,:,3) = fnoise_B;
            fnoise2 = imresize(fnoise2,[h w]);
            fnoise2 = fnoise2*255;
            
            wei = pink_level(level);
            
            distorted_img = double(img) + wei*fnoise2;
            distorted_img = floor(distorted_img);
            distorted_img = uint8(distorted_img);
%         case 27 % Chromatic aberrations
%             gblur_level = [1,1,1,1,6];
%             hsize = gblur_level(level);
%             %  hsize=3;
%             level = ca_level(level);
%             R=(img(:,:,1));
%             G=(img(:,:,3));
%             B=(img(:,:,2));
%             R2=R;
%             B2=B;
%             R2(:,level:end)=R(:,1:end-level+1);
%             B2(:,round(level/2):end)=B(:,1:end-round(level/2)+1);
%             temp = img;
%             temp(:,:,1)=R2;
%             temp(:,:,2)=B2;
%             distorted_img=temp;
%             h = fspecial('gaussian', hsize, hsize/6);
%             distorted_img=imfilter(distorted_img,h,'symmetric');

        case 28 %over-Exposure
            distorted_img = simulate_ou(img, 1, oe_level(level));
            
        case 29 %under-Exposure
            distorted_img = simulate_ou(img, 2, ue_level(level));
    end
end


