function pprint(x)

for i=1:len(x)
    if iscell(x)
        disp(x{i});
    else
        disp(x(i,:));
    end
end