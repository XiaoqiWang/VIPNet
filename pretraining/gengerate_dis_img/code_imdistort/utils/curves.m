function yy = curves(xx,coef)

xx = double(xx);

defa = 1;

if isscalar(coef)
    coef = [0.5; coef]; 
    defa = 0;
elseif len(coef(:))==2
    coef = [0.3 0.5 0.7;
            coef(1) 0.5 coef(2)];
    defa = 0;
end

if ~defa
    x = [0 coef(1,:) 1];
    y = [0 coef(2,:) 1];
else
    x = coef(1,:);
    y = coef(2,:);
end

cs = spline(x,y);
yy = ppval(cs,xx);
yy = max(0,min(1, yy));
