function NMSE_calc = NMSE( wb, net, input, target)

% wb is the weights and biases row vector.

% It must be transposed when transferring the weights and biases to the network net.

 net = setwb(net, wb');

% The net output matrix is given by net(input). The corresponding error matrix is given by
 output = net(input);
 error = abs(vec2ind(target') - vec2ind(output));

% The mean squared error normalized by the mean target variance is

 NMSE_calc = mean(error.^2)/mean(var(vec2ind(target'),1));% + 0.1*sum(abs(wb));

 
% It is independent of the scale of the target components and related to the Rsquare statistic via

% Rsquare = 1 - NMSEcalc ( see Wikipedia)
