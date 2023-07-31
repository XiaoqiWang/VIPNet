function save_true = ChromaticAberrations(img,img_name,dist_type, target_path)
strs = split(img_name,'.');
level_1=0;level_2 =0; level_3 =0; level_4=0; level_5=0;
gblur_level = [1,2,3,4,5,7,9,15,20,25,30];
gap_33db = 2;
gap_30db = 2;
gap_27db = 2;
gap_24db = 2;
gap_21db = 2;
for i=1:25
    for l = 1:11
        distorted_img = CA(img,i,gblur_level(l));
        PSNR = psnr(img,distorted_img);
        if PSNR>33 || abs(PSNR-33)<gap_33db
            distorted_img_1 = distorted_img;
            if l<=2 && i<=4
                distorted_img_1 =distorted_img;
                gap_33db = abs(PSNR-33);
                level_1 =1;
            end
        elseif abs(PSNR-30)<gap_30db
            if l>=2&&l<=5 && i<=20
                distorted_img_2 =distorted_img;
                gap_30db = abs(PSNR-30);
                level_2 =1;
            end
        elseif abs(PSNR-27)<gap_27db
            if l>=3 &&l<=6 && i<=20
                distorted_img_3 =distorted_img;
                gap_27db = abs(PSNR-27);
                level_3 =1;
            end
        elseif abs(PSNR-24)<gap_24db
            if l>=3&&l<=8 && i>=5 && i<=20
                distorted_img_4 =distorted_img;
                gap_24db = abs(PSNR-24);
                level_4 =1;
            end
        elseif abs(PSNR-21)<gap_21db
            if l>=5&&l<=11 && i>=7 && i<=25
                distorted_img_5 =distorted_img;
                gap_21db = abs(PSNR-21);
                level_5 =1;
            end
        end
    end
    
    
end
if level_1 && level_2 && level_3 && level_4 && level_5 ~=0
    imwrite(distorted_img_1,[target_path , strs{1}  '_' num2str(dist_type,'%02d')  '_01' ,'.bmp'])
    imwrite(distorted_img_2,[target_path , strs{1}  '_' num2str(dist_type,'%02d')  '_02' ,'.bmp']);
    imwrite(distorted_img_3,[target_path , strs{1}  '_' num2str(dist_type,'%02d')  '_03' ,'.bmp']);
    imwrite(distorted_img_4,[target_path , strs{1}  '_' num2str(dist_type,'%02d')  '_04' ,'.bmp']);
    imwrite(distorted_img_5,[target_path , strs{1}  '_' num2str(dist_type,'%02d')  '_05' ,'.bmp']);
    save_true=1;
else
    save_true=0;
end
end

 function b= CA(img_rgb,level, hsize)
 
%  hsize=3;

 R=(img_rgb(:,:,1));
 G=(img_rgb(:,:,3));
 B=(img_rgb(:,:,2));
 R2=R;
 B2=B;
 R2(:,level:end)=R(:,1:end-level+1);
 B2(:,level/2:end)=B(:,1:end-level/2+1);
 temp = img_rgb;
 temp(:,:,1)=R2;
 temp(:,:,2)=B2;
 img=temp;
 h = fspecial('gaussian', hsize, hsize/6);
 b=imfilter(img,h,'symmetric');

end