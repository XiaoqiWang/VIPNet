function J = imgrains(I, amount, alpha)

if ~exist('alpha', 'var')
    alpha = rand()/2 + 0.5;
end

J = I;
if nchans(J)==1
    J = repmat(J, [1 1 3]);
end

% J = imgrain(J, amount, 2);

a = [amount.^(1-alpha) (7*amount).^alpha];
a = 4*amount .* (a./sum(a)).^2;

J = imgrain(J, a(1), 0);
J = imgrain(J, a(2), 1);

