function [J, type, amount] = imdistort(I, fn, range, perc)

if ~exist('perc', 'var')
        perc = rand();
end

if iscell(fn)
    if ~exist('range','var')
        perc = rand();
    else
        perc = range;
    end
    range = fn{2};
    fn = fn{1};
end

amount = diff(range) * perc + range(1);

J = truncate( fn(I, amount) );

type = char(fn);

% fprintf('@%s %g\n', type, amount);