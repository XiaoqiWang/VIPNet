function [x minx maxx] = mapmm_fast(x,mina,maxa)  %#codegen
%MAPMM Map any array's minimum and maximum values to [0,1] or specified values.
%    
%  Syntax
%  
%   [a mindif maxdif] = mapminmax(a)
%   [a mindif maxdif] = mapminmax(a,mina,maxa)
%
%  Description
%    
%   MAPMM processes arrays (any shape) by normalizing the global 
%   minimum and maximum values to [MINA, MAXA] or [0,1] if not specified.
%    
%   It is assumed that A has only finite real values.
%  
%  See also MAPMINMAX

if nargin<2
    mina = 0.;
    maxa = 1.;
else
    maxa = double(maxa);
    mina = double(mina);
end

 
minx = min2(x);
maxx = max2(x);
if maxx>minx
    x = (x-minx)/(maxx-minx)*(maxa-mina)+mina;
end