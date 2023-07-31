%% setup
clear; clc;
warning off
addpath(genpath('code_imdistort'));


%% read the info of pristine images
sourcePath = 'F:/kadis700k/kadis700k/';
targetPath = 'E:/wxq/';

tb = readtable('./dataset_ref_img.csv');
tb = table2cell(tb);

%% generate distorted images in dist_imgs folder

 for i = 1:size(tb,1)
     img_idx = sprintf('%03d', ceil(i/800));
     ref_im = imread([sourcePath, 'ref_imgs/' tb{i,1}]);
     for dist_type = 26:29
         if dist_type==27
             continue
         end
         for dist_level = 1:5
             [dist_im] = additional_distortion_generator(ref_im, dist_type, dist_level);
             strs = split(tb{i,1},'.');
             dist_im_name = [strs{1}  '_' num2str(dist_type,'%02d')  '_' num2str(dist_level,'%02d') '.bmp'];
             disp(dist_im_name);
             imwrite(dist_im, [targetPath, 'pretrained_data','/', img_idx, '/', dist_im_name]);
         end

     end

 end

tb = readtable('./dataset_ref_img_140k.csv');
tb = table2cell(tb);
dist_type=27;
img_num = 0;
for i = 1:size(tb,1)
    ref_im = imread([sourcePath, 'ref_imgs/' tb{i,1}]);
    if img_num==40000
        break
    end
    img_idx = sprintf('%03d', ceil((img_num+1)/800));
    save_true = ChromaticAberrations(ref_im, tb{i,1}, dist_type, [targetPath,'pretrained_data','/', img_idx, '/']);
    img_num=img_num+save_true;  
    disp(img_num)
end







