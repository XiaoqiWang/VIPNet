function J = impixelate(I, strength)

z = 0.95 - strength.^0.6;
Jlo = imresize(I, z, 'nearest');
J = imresize(Jlo, sizeim(I), 'nearest');
