function J = imsharpenLo(I, amount)

J = imoverlay(I, amount * imdog(I, 0.7, 30) + 0.5);
J = truncate(J);