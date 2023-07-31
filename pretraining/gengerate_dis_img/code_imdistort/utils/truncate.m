function x = truncate(x,mi,ma)
% function x = truncate(x,mi,ma)
% Truncate to 0,1 by default

if nargin<3, ma = 1; end
if nargin<2, mi = 0; end

x(:) = max(min(x(:),ma),mi);
