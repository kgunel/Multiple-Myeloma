clear; close; clc
%% Problem Definition
method = 'Adaptive Nelder-Mead with Weighted Centroids';
% Inputs for Multiple Myeloma Tian Data Set
% contains features to predict 
% whether a patient contains signs of multiple myeloma or not.
load('data/multiple_myeloma.mat')

%% Parameters
m = [2 3 5];                % number of neurons
n = length(inputs(:,1));    % # of features
N = length(targets(:,1));   % # of inputs 
params.low = -100;          % Lower Bound of Variables
params.up = 100;            % Upper Bound of Variables
params.MaxIt = 100;         % Maximum Number of Iterations
params.maxnFeval = 10^6;
params.eps = 1e-6;

%% Normalize input data 
% Z-score normalization
%Z = normalize(inputs,2);
% PCA
%[coeff,score,latent] = pca(inputs');
%Xcentered  = score*coeff';
%inputs = Xcentered';
%inputs = inputs(1:N-1,:);

%% Feature Importance using Minimum Redundancy Maximum Relevance (MRMR) algorithm
%targets = 1 - (targets(:,1)==1);
%targets(targets==0)=2;
%targets = targets';
% [idx,scores] = fscmrmr(inputs',targets);
% inputs = inputs(idx(1:81),:);
load('data/selected_features.mat')
targets = onehotencode(categorical(targets),2);

%% Creating Data Partition using k-fold Cross-Validition
dataIndices = 1:N;
K = 10;
indices = crossvalind('Kfold',dataIndices,K);
% Initialize an object to measure the performance of the classifier.
cp = classperf(targets);

%% Creating a neural network
net = patternnet(m);
net = init(net);
% configure the neural network for this dataset
net = configure(net, inputs, targets');
% get the normal NN weights and bias
wb = getwb(net);
params.dim = length(wb); % # of neural net parameters

wb = params.low + (params.up - params.low)*rand(1,params.dim);
net = setwb(net, wb);
% k-fold Cross-Validation
T = zeros(1,K);
for k = 1:K
    fprintf('\nTrial Id  %2d\n',k);
    test = (indices == k);
    train = ~test;
      
    X = inputs(:,train);
    Y = targets(train,:);

    % create handle to the MSE_TEST function, that calculates MSE
    Cost = @(x) NMSE(x, net, X, Y);
    
    % running the algorithm with desired options
    tic
    [Sol(k).xmin,fmin(k),nFeval(k),nRef(k),nExp(k), nIC(k), nOC(k), nShrink(k), nIter(k),Sol(k).BestCost] = wANMS(Cost,wb,params);
    wb = Sol(k).xmin;
    net = setwb(net, wb');
    T(k) = toc;
    outs = net(X);
    Sol(k).results = calculateResults(net,Y,outs,method,T(k),k);
    Sol(k).inputs = X;
    Sol(k).targets = Y;
    if k==1
        bestsofar = fmin(k);
        bestofBestCost = Sol(k).BestCost;
        bestSol = Sol(k);
        bestWeights = Sol(k).xmin;
    else
        if fmin(k) < bestsofar
            bestsofar = fmin(k);
            bestofBestCost = Sol(k).BestCost;
            bestSol = Sol(k);
            bestWeights = Sol(k).xmin;
        end
    end
    % get the optimized NN weights and bias
    wb = getwb(net);
end

%% Best Solution
net = setwb(net, bestWeights');
bestSol.outs = net(bestSol.inputs);

%% Display Results
% for best solution
fprintf('----------------------------------------------------------------------\n');
fprintf('Results for Best Solution :\n');
fprintf('\tMSE = %f\n',bestSol.results.MSE);
fprintf('\tR^2 = %f\n',bestSol.results.Rsquare);
fprintf('\tAUC = %f\n',bestSol.results.AUC);
fprintf('\tAccuracy = %f\n', bestSol.results.Accuracy);
fprintf('\tSensitivity or Recall = %f\n', bestSol.results.Sensitivity);
fprintf('\tSpecificity = %f\n', bestSol.results.Specificity);
fprintf('\tPrecision = %f\n', bestSol.results.Precision);
fprintf('\tf1-score = %f\n', bestSol.results.f1_score);
fprintf('\tMatthews correlation coefficient (MCC) = %f\n', bestSol.results.MCC);
fprintf('\tJaccard index = %f\n', bestSol.results.Jaccard_ind);
fprintf('\tCohen''s Kappa coefficient = %f\n', bestSol.results.kappa);
fprintf('----------------------------------------------------------------------\n');

fid=fopen(['data\Results_' method '.txt'],'a+');
fprintf(fid,'\nResults for Best Solution :\n');
fprintf(fid,'\tMSE = %f\n',bestSol.results.MSE);
fprintf(fid,'\tR^2 = %f\n',bestSol.results.Rsquare);
fprintf(fid,'\tAUC = %f\n',bestSol.results.AUC);
fprintf(fid,'\tAccuracy = %f\n', bestSol.results.Accuracy);
fprintf(fid,'\tSensitivity or Recall = %f\n', bestSol.results.Sensitivity);
fprintf(fid,'\tSpecificity = %f\n', bestSol.results.Specificity);
fprintf(fid,'\tPrecision = %f\n', bestSol.results.Precision);
fprintf(fid,'\tf1-score = %f\n', bestSol.results.f1_score);
fprintf(fid,'\tMatthews correlation coefficient (MCC) = %f\n', bestSol.results.MCC);
fprintf(fid,'\tJaccard index = %f\n', bestSol.results.Jaccard_ind);
fprintf(fid,'\tCohen''s Kappa coefficient = %f\n', bestSol.results.kappa);
fprintf(fid,'----------------------------------------------------------------------\n');


% for Average solutions
rslts = [Sol(:).results];
MSEs = [rslts.MSE];
Rsquares = [rslts.Rsquare];
AUCs = [rslts.AUC];
Specificitives = [rslts.Specificity];
Accuracies = [rslts.Accuracy];
Sensitivities = [rslts.Sensitivity];
Precisions = [rslts.Precision];
f1_scores = [rslts.f1_score];
MCCs = [rslts.MCC];
Jaccards = [rslts.Jaccard_ind];
kappas = [rslts.kappa];
fprintf('\nAverage Results\n');
fprintf('\tMean of MSEs = %5.3f %c %5.3f\n',mean(MSEs),char(177),std(MSEs));
fprintf('\tMean of R^2s = %5.3f %c %5.3f\n',mean(Rsquares),char(177),std(Rsquares));
fprintf('\tMean of AUCs = %5.3f %c %5.3f\n',mean(AUCs),char(177),std(AUCs));
fprintf('\tMean of Accuracies = %5.3f %c %5.3f\n', mean(Accuracies),char(177),std(Accuracies));
fprintf('\tMean of Sensitivities or Recalls = %5.3f %c %5.3f\n', mean(Sensitivities),char(177),std(Sensitivities));
fprintf('\tMean of Specificitives = %5.3f %c %5.3f\n', mean(Specificitives),char(177),std(Specificitives));
fprintf('\tMean of Precisions = %5.3f %c %5.3f\n', mean(Precisions),char(177),std(Precisions));
fprintf('\tMean of f1-scores = %5.3f %c %5.3f\n', mean(f1_scores),char(177),std(f1_scores));
fprintf('\tMean of  Matthews correlation coefficients (MCCs) = %5.3f %c %5.3f\n', mean(MCCs),char(177),std(MCCs));
fprintf('\tMean of Jaccard indexes = %5.3f %c %5.3f\n', mean(Jaccards),char(177),std(Jaccards));
fprintf('\tMean of Cohen''s Kappa coefficients = %5.3f %c %5.3f\n', mean(kappas),char(177),std(kappas));
fprintf('Mean of Elapsed Time : %5.3f %c %5.3f seconds\n',mean(T),char(177),std(T));
fprintf('----------------------------------------------------------------------\n');


fprintf(fid,'\nAverage Results\n');
fprintf(fid,'\tMean of MSEs = %5.3f %c %5.3f\n',mean(MSEs),char(177),std(MSEs));
fprintf(fid,'\tMean of R^2s = %5.3f %c %5.3f\n',mean(Rsquares),char(177),std(Rsquares));
fprintf(fid,'\tMean of AUCs = %5.3f %c %5.3f\n',mean(AUCs),char(177),std(AUCs));
fprintf(fid,'\tMean of Accuracies = %5.3f %c %5.3f\n', mean(Accuracies),char(177),std(Accuracies));
fprintf(fid,'\tMean of Sensitivities or Recalls = %5.3f %c %5.3f\n', mean(Sensitivities),char(177),std(Sensitivities));
fprintf(fid,'\tMean of Specificitives = %5.3f %c %5.3f\n', mean(Specificitives),char(177),std(Specificitives));
fprintf(fid,'\tMean of Precisions = %5.3f %c %5.3f\n', mean(Precisions),char(177),std(Precisions));
fprintf(fid,'\tMean of f1-scores = %5.3f %c %5.3f\n', mean(f1_scores),char(177),std(f1_scores));
fprintf(fid,'\tMean of  Matthews correlation coefficients (MCCs) = %5.3f %c %5.3f\n', mean(MCCs),char(177),std(MCCs));
fprintf(fid,'\tMean of Jaccard indexes = %5.3f %c %5.3f\n', mean(Jaccards),char(177),std(Jaccards));
fprintf(fid,'\tMean of Cohen''s Kappa coefficients = %5.3f %c %5.3f\n', mean(kappas),char(177),std(kappas));
fprintf(fid,'\nMean of Elapsed Time : %5.3f %c %5.3f seconds\n',mean(T),char(177),std(T));
fprintf(fid,'----------------------------------------------------------------------\n');
fclose(fid);

%% Plots
newcolors = {'#F00','#F80','#FF0','#0B0','#00F','#50F','#A0F'};
clrord = colororder(newcolors);
% Plot Cost 
figure
semilogy(bestofBestCost,'LineWidth',1.5);
title('Cost Function','FontSize',12,'FontWeight','b');
xlabel('Iteration','FontSize',12,'FontWeight','b');
ylabel('Fitness(Best-so-far)','FontSize',12,'FontWeight','b');
savefig('figs\cost.fig')
print(gcf,'figs\cost','-depsc','-r300')
print(gcf,'figs\cost','-dpng','-r300')

% Plot Confusion Matrix
%figure, cm = confusionchart(vec2ind(targets'),vec2ind(outs));
%figure, cm = confusionchart(bestSol.results.cm);
figure
classLabels ={'Positive'; 'Negative'};
cm = confusionchart(bestSol.results.ConfusionMatrix,classLabels,...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'DiagonalColor',[0.3010 0.7450 0.9330],...
    'OffDiagonalColor', [1 1 1] );
cm.Title = {'Classification of Tian et al.''s Multiple Myeloma Dataset '
            ['using Multi-layer Perceptron trained by ' method]};
sortClasses(cm,'descending-diagonal');
cm.Normalization = 'absolute';
savefig('figs\cm.fig')
print(gcf,'figs\cm','-depsc','-r300')
print(gcf,'figs\cm','-dpng','-r300')

% Plot ROC Curve
[tpr,fpr,thresholds] = roc(bestSol.targets',bestSol.outs);
figure
hold on
plot(fpr{1},tpr{1},'Color',[0 0 1],'LineStyle','-','LineWidth',2,'Marker','s','MarkerIndices',1:5:length(tpr{1}));
plot(fpr{2},tpr{2},'Color',[0 0.7333 0],'LineStyle','-.','LineWidth',2,'Marker','d','MarkerIndices',1:5:length(tpr{2}));
legend({'Positive','Negative'} ,'Location','southeast','AutoUpdate','off'); 
plot([0 1], [0 1], 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
hold off
savefig('figs\roc.fig')
print(gcf,'figs\roc','-depsc','-r300')
print(gcf,'figs\roc','-dpng','-r300')

%% Saving Results
save(['data\Results_' method '.mat']);