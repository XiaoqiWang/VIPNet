function a = mapcells(fn,c,b,varargin)
%MAPCELLS mapcell with 2 arguments.

    a = cell(size(c));
    for i=1:numel(c)
        a{i} = feval(fn,c{i}, b{i}, varargin{:});
    end