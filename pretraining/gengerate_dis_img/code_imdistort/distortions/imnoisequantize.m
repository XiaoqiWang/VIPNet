function J = imnoisequantize(I,levels)

threshRGB = multithresh(I,levels);

threshForPlanes = zeros(3,levels);			

for i = 1:3
    threshForPlanes(i,:) = multithresh(I(:,:,i),levels);
end

value = [0 threshRGB(2:end) 1];
J = imquantize(I, threshRGB, value);

end