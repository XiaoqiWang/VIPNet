function r = len(a) %#codegen
%LEN Alias for length.
%
%  See also LENGTH

if istable(a)
    r = size(a,1);
else
    r = length(a);
end