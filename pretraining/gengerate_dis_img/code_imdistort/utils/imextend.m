function b = imextend(a, newsz)

newsz = fliplr(newsz);

if nchans(a)>1
    sz = [newsz 1];
else
    sz = newsz;
end

b = extend(a, sz, 'symmetric');
b = double(b);



