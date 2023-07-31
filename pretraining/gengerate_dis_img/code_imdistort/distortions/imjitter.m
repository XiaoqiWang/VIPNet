function J = imjitter(I, amount)
I = im2double(I);
J = imscatter(I, [], amount, 5);