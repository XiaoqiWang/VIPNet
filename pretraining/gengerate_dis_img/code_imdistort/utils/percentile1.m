function v = percentile1(x, perc)

x = double(x);
xs = sort(x(:));
i = len(xs)*perc/100.;
i = truncate(i, 1, len(xs));
v = xs(round(i));