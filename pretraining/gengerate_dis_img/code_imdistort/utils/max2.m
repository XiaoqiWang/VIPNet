function maxx = max2(x)  %#codegen
% MAX2 Largest value.
%
%   Returns the global maximum of any array shape.

maxx = x;
for i=1:ndims(x)
    maxx = max(maxx);
end
maxx = maxx(1);