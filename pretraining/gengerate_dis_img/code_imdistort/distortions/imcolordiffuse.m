function J = imcolordiffuse(I, amount)

J = imcolorblur(I, 1.5*amount + 2, amount);