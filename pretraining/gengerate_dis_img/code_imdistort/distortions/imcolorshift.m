function J = imcolorshift(I, amount)

I = im2double(I);
x = rgb2gray(I);
e = gradmag1(x);
e = mapmm(imgaussfilt(e, 4));
e = ps2(truncate(e, 0.1, 1));

% channel = randi(3,1);
channel = 2;
g = I(:,:,channel);
amount_shift = round(normr(rand(1,2))*amount);
g = imshift(g, amount_shift);

J = I;
J(:,:,channel) = imblend(g, I(:,:,channel), e);