function [b] = immeanshift(a,amount)


b = truncate(a+amount,0,1);

end