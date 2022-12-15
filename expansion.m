function E = expansion(beta,M,R)
%   n = length(beta);
%   nVar = length(M);
%   E =zeros(n,nVar);
%   for i=1:n
%      %E(i,:) = M + beta(i)*(R - M);
%      E(i,:) = M + beta.*(R - M);
%   end
E = M + beta.*(R - M);
end