function Igrain = imgrain(I, NoiseVar, BlurVar)

if ~exist('BlurVar','var')
    BlurVar = 0;
end

if nchans(I)==3
    lab = rgb2lab(I);
    L = lab(:,:,1)/100;
else
    L = I;
end

N = randn(size(L)) * NoiseVar;

if BlurVar
    N = imgaussfilt(N, BlurVar);
end

Lgrain = truncate(L + N.*(1-L));

if nchans(I)==3
    lab(:,:,1) = Lgrain*100;
    Igrain = truncate(lab2rgb(lab));
else
    Igrain = Lgrain;
end