function y = imshift(x, shift)

padding = max(shift)*[1 1];

y = padarray(x, padding, 'replicate', 'both');
y = circshift(y, shift);
y = unpadarray(y, padding, 'both');
