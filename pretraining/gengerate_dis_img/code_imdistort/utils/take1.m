function y = take1(x, n)
% function y = take1(x, n)

if ~exist('n','var')
    n = 1;
end
if n<0
    n = len(x)+n+1;
end

if iscell(x)
    y = x{n};
elseif istable(x)
    y = x{1,1};
else
    y = x(n);
end