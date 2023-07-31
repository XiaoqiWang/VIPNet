function b = imnoisegauss(a,var)

b = imnoise(a,'gaussian', 0, var);