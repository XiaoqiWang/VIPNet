function n = nchans(x)
if iscell(x)
    n = len(x);
else
    n = size(x,3);
end