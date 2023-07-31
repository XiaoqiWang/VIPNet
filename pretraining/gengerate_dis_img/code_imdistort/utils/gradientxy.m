function r = gradientxy(varargin)

[x, y] = gradient(varargin{:});
r = cat(3,x,y);