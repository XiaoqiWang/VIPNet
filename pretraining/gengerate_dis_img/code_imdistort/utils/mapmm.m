function [a minx maxx] = mapmm(a,mina,maxa) %#codegen
%MAPMM Map any array's minimum and maximum values to [0,1] or specified values.
%    
%  Syntax
%  
%   [a mindif maxdif] = mapminmax1(a)
%   [a mindif maxdif] = mapminmax1(a,mina,maxa)
%
%  Description
%    
%   MAPMM processes arrays (any shape) by normalizing the global 
%   minimum and maximum values to [MINA, MAXA] or [0,1] if not specified.
%    
%   It is assumed that A has only finite real values.
%  
%  See also MAPMINMAX

if ~isa(a,'dip_image')
    x = double(a(:));
else
    x = a;
end

if nargin<2
    mina = 0.;
    maxa = 1.;
else
    maxa = double(maxa);
    mina = double(mina);
end

minx = min(x);
maxx = max(x);
if minx<maxx
    x = (x-minx)/(maxx-minx)*(maxa-mina)+mina;
    if ~isa(a,'dip_image')
        a = reshape(x,size(a));
    else
        a = x;
    end
end