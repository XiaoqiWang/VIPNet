%reference：
%@ARTICLE{9548827, 
%author={Ou, Fu-Zhao and Wang, Yuan-Gen and Li, Jin and Zhu, Guopu and Kwong, Sam},  
%journal={IEEE Transactions on Multimedia},   
%title={A Novel Rank Learning Based No-Reference Image Quality Assessment Method},   year={2021},  volume={},  number={},  pages={1-1},  
%doi={10.1109/TMM.2021.3114551}}
function output = simulate_ou(temp_img, type, level)

switch type
    case 1
        [m,n,k1] = size(temp_img);
        hsv = rgb2hsv(temp_img);
        V = hsv(:,:,3);
 
        hsv = rgb2hsv(temp_img);

        for j = 1:m   
            for k = 1: n

        sharp = 0.006 * hsv(j,k,3)^3.00;
        sharp2 = 0.0315*hsv(j,k,3)^2.42;
        hsv(j,k,3) = sharp*level^2+sharp2*level+(hsv(j,k,3));

            end
        end
        output = im2uint8(hsv2rgb(hsv));

    case 2
              
        [m,n,k1] = size(temp_img);
        hsv = rgb2hsv(temp_img);
        V = hsv(:,:,3);
 
        hsv = rgb2hsv(temp_img);

        for j = 1:m   
            for k = 1: n

        sharp = -0.0018 * log(hsv(j,k,3))-0.00005;
        sharp2 = 0.000021 * hsv(j,k,3)^(-3.0);
        hsv(j,k,3) = -sharp*level^2-sharp2*level+(hsv(j,k,3));

            end
        end
        output = im2uint8(hsv2rgb(hsv));
end