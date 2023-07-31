function b = imblur(a, type, radius, angle)

if strcmp(type,'lens')
    
    k = fspecial('disk', radius);
    b = imfilter(a, k, 'replicate','same');
%     ae = padarray(a, [radius radius], 'replicate', 'both');
%     b = mapim(@conv2,ae,k,'valid');
    
elseif strcmp(type,'motion')
    
    if ~exist('angle','var')
        angle = randi([0, 180], 1);
    end
    
    k = fspecial('motion', radius, angle);
    b = imfilter(a, k, 'replicate','same');
%     ae = padarray(a, floor(size(k)/2), 'replicate', 'both');
%     b = mapim(@conv2,ae,k,'valid');  
else 
    
    b = imgaussfilt(a, radius);
    
end