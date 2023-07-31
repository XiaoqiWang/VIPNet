function J = imnoisemultiplicative(I,var)

J = imnoise(I,'speckle',var);

end