function b =  unpadarray(a, padsize, type)

if ~exist('type', 'var')
    type = 'both';
end

p = padsize(1);
q = padsize(2);

if len(padsize)==2
    if strcmp(type,'both')
        b = a(p+1:end-p, q+1:end-q,:);
    elseif strcmp(type,'pre')
        b = a(p+1:end, p+1:end,:);
    else
        b = a(1:end-q, 1:end-q,:);
    end
elseif len(p)==4
    pend = padsize(3);
    qend = padsize(4);

    b = a(p+1:end-pend, q+1:end-qend,:);
else
    error('padsize must be a 2 or 4 element vector');
end