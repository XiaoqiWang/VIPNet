function a = mapcell(fn,c,varargin)
%MAPCELL Iteratively map a function on each element in a cell array.
%
%  out = mapcell(fn,cells)
%
%  Example
%
%    Increment all values in a cell array by 1:
%
%     c = num2cell(1:3)
%
%     c = 
%
%       [1]    [2]    [3]
%
%     mapcell(@(x)x+1,c)
%
%     ans = 
%
%       [2]    [3]    [4]
%
    a = cell(size(c));
    
    p = gcp('nocreate');
    
    if ~isempty(p)
        parfor i=1:numel(c)
            a{i} = feval(fn,c{i},varargin{:});
        end
    else
        for i=1:numel(c)
            a{i} = feval(fn,c{i},varargin{:});
        end
    end