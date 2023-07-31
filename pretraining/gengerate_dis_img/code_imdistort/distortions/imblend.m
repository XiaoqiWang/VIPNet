function z = imblend(x, y, w)

z = immultiply(x,w) + immultiply(y, 1-w);