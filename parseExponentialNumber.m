function str = parseExponentialNumber(str,d,flag,tol)
% if flag = 0, it means no overflow or underflow error occured
% flag = 1 means overflow
% flag = -1 means underflow
ind = strfind(str,'e');
if ~isempty(ind)
    realPart = str(1:ind-1);
    ind2 = strfind(realPart,'.');
    if ~isempty(ind2)
        if (ind2+d) < length(realPart)
            realPart = realPart(1:ind2+d); % d ondalýklý reel sayý
        end
    end
    
    if flag == 1
        exponentialPart = num2str(str2double(str(ind+1:end)) + tol);
    elseif flag == -1
        exponentialPart = num2str(str2double(str(ind+1:end)) - tol);
    else
        exponentialPart = str(ind+1:end);
    end
    str = [realPart '$\times 10^{' exponentialPart '}$'];
else
    ind2 = strfind(str,'.');
    if ~isempty(ind2)
        if (ind2+d) < length(str)
            str = str(1:ind2+d); % d ondalýklý reel sayý
        end
    end
end
end