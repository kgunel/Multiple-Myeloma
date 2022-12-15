clear; close; clc;
csvInputFile = 'data/tian_inputs.csv';
opts = detectImportOptions(csvInputFile);
%preview(csvInputFile,opts)
X = readmatrix(csvInputFile,opts);
inputs = X';
csvOutputFile = 'data/tian_outputs.csv';
opts = detectImportOptions(csvOutputFile);
Y = readmatrix(csvOutputFile,opts);
targets = onehotencode(categorical(Y),2);

%% Feature Importance using Minimum Redundancy Maximum Relevance (MRMR) algorithm
targets = 1 - (targets(:,1)==1);
targets(targets==0)=2;
targets = onehotencode(categorical(targets),2);
[idx,scores] = fscmrmr(inputs',targets);
tol = 1e-3;

inputs = inputs(idx(1:81),:);


save('data/multiple_myeloma.mat','inputs','targets');
