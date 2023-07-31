function J = imscatter(I, kern, amount, iterations)

if ~exist('iterations', 'var')
    iterations = 1;
end

sz = [sizeim(I), 2];  

J = I;
for i = 1:iterations
    shifts = randn(sz)*amount;

    if ~isempty(kern)
        shifts = imdilate(shifts, kern);
        shifts = zscore(shifts)*amount;
    end

    J = imwarpmap(J, shifts, 'bicubic', true);
end