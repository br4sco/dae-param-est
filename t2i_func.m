% In the vector [xN, ..., x1]^T, t21([t1,..,t1]) gives the row indices of 
% xt1, ..., xtq ordered such that the ordering of the elements is the same
% as in the original vector

function [rows] = t2i_func(time, N, n)

    rows = nan(length(time)*n, 1);
    j = 0;
    for t=time
        rows( (j*n+1):((j+1)*n), :) = ((N-t)*n + 1):((N-t+1)*n);
        j = j + 1;
    end
    rows = sort(rows);
end

