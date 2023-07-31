function J = imcontrastc(I, amount)
% amount in [-1, 1]

J = curves(I, [0.25-amount/4 0.75+amount/4]);