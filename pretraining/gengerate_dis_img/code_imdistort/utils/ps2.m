function b = ps2(a, percdev)
% b = ps2(a, percdev)

if ~exist('percdev','var')
    percdev = [1 1];
elseif isscalar(percdev)
	percdev = [percdev percdev];
end

a = mapmm(a);

valuehi = percentile1(a,100-percdev(2));
valuelo = 1-percentile1(1-a,100-percdev(1));

b = max(min(a, valuehi), valuelo);
b = mapmm(double(b));