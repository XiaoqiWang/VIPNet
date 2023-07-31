function J = imnoiseimpulse(I,d)

J = imnoise(I,'salt & pepper',d);

end