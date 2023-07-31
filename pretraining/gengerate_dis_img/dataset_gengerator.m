%% setup
clear; clc;
addpath(genpath('code_imdistort'));


%% read the info of pristine images

sourcePath = 'F:/kadis700k/kadis700k/';
targetPath = 'E:/wxq/';

mkdir(targetPath,'pretrained_data')
 for i =1:50
     num = sprintf('%03d', i);
     mkdir([targetPath, 'pretrained_data','/'],num2str(num));
 end
 
tb = readtable('./dataset_ref_img.csv');
tb = table2cell(tb);

%% generate distorted images in dist_imgs folder

for i = 1:size(tb,1)
    img_idx = sprintf('%03d', ceil(i/800));
    ref_im = imread([sourcePath, 'ref_imgs/' tb{i,1}]);
    for dist_type = 1:25
    %dist_type = tb{i,2};
        for dist_level = 1:5
            [dist_im] = imdist_generator(ref_im, dist_type, dist_level);
            strs = split(tb{i,1},'.');
            dist_im_name = [strs{1}  '_' num2str(dist_type,'%02d')  '_' num2str(dist_level,'%02d') '.bmp'];
            disp(dist_im_name);
            imwrite(dist_im, [targetPath, 'pretrained_data','/', img_idx, '/', dist_im_name]);
        end
        
    end 
    
end







