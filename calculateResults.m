function results = calculateResults(net,targets,outs,method,T,trialId)

targets = targets';
results.T = T;
results.cm = confusionchart(vec2ind(targets),vec2ind(outs));
[results.tpr,results.fpr,thresholds] = roc(targets,outs);

%perf = perform(net,outs,targets);
error = abs(vec2ind(targets) - vec2ind(outs));
results.MSE = mean(error.^2);
results.NMSE = mean(error.^2)/mean(var(targets,1));
results.Rsquare = 1 - results.NMSE;

[X,Y,thresholds,results.AUC] = perfcurve(vec2ind(targets),vec2ind(outs),1);
results.p = sum(results.cm.NormalizedValues(1,:));  % Positive samples
results.n = sum(results.cm.NormalizedValues(2,:));  % Negative samples
results.pp = sum(results.cm.NormalizedValues(:,1)); % Positive predictions
results.pn = sum(results.cm.NormalizedValues(:,2)); % Negative predictions
results.tp = results.cm.NormalizedValues(1,1);
results.tn = results.cm.NormalizedValues(2,2);
results.fp = results.cm.NormalizedValues(2,1);
results.fn = results.cm.NormalizedValues(1,2);

results.Accuracy = (results.tp+results.tn)/(results.p + results.n);
results.Sensitivity = results.tp/results.p;     % Recall or True positive rate (TPR)
results.Specificity = results.tn/results.n;     % True negative rate (TNR)
results.Precision = results.tp/results.pp;      % Positive predictive value (PPV)
results.f1_score = 2*results.tp/(2*results.tp+results.fp+results.fn);   % f-measure

results.tpr = results.Sensitivity;      % True positive rate (TPR)
results.tnr = results.Specificity;      % True negative rate (TNR)
results.fpr = results.fp/results.n;     % False positive rate (FPR)
results.fnr = results.fn/results.p;     % False negative rate (FNR)
results.ppv = results.Precision;        % Positive predictive value (PPV)
results.npv = results.tn/results.pn;    % Negative predictive value (NPV)
results.fOr = results.fn/results.pn;    % False omission rate (FOR)
results.fdr = results.fp/results.pp;    % False discovery rate (FDR)
results.MCC = sqrt(results.tpr*results.tnr*results.ppv*results.npv) - sqrt(results.fpr*results.fnr*results.fOr*results.fdr); % Matthews correlation coefficient (MCC)
results.Jaccard_ind = results.tp/(results.tp+results.fn+results.fp);    % Jaccard index 
results.kappa = 2*(results.tp*results.tn - results.fn*results.fp)/((results.tp+results.fp)*(results.fp+results.tn) + (results.tp+results.fn)*(results.fn+results.tn)); % Cohen's Kappa coefficient

results.ConfusionMatrix = results.cm.NormalizedValues;
fprintf('\t\tResults for Trial Id : %d\n',trialId);
fprintf('\t\t\tMSE = %f\n',results.MSE);
fprintf('\t\t\tR^2 = %f\n',results.Rsquare);
fprintf('\t\t\tAUC = %f\n',results.AUC);
fprintf('\t\t\tAccuracy = %f\n', results.Accuracy);
fprintf('\t\t\tSensitivity or Recall = %f\n', results.Sensitivity);
fprintf('\t\t\tSpecificity = %f\n', results.Specificity);
fprintf('\t\t\tPrecision = %f\n', results.Precision);
fprintf('\t\t\tf1-score = %f\n', results.f1_score);
fprintf('\t\t\tMatthews correlation coefficient (MCC) = %f\n', results.MCC);
fprintf('\t\t\tJaccard index = %f\n', results.Jaccard_ind);
fprintf('\t\t\tCohen''s Kappa coefficient = %f\n', results.kappa);

if trialId ==1
    fid=fopen(['data\Results_' method '.txt'],'w+');
else
    fid=fopen(['data\Results_' method '.txt'],'a+');
end
fprintf(fid,'\nResults for Trial Id : %d\n',trialId);
fprintf(fid,'\t\tMSE = %f\n',results.MSE);
fprintf(fid,'\t\tR^2 = %f\n',results.Rsquare);
fprintf(fid,'\t\tAUC = %f\n',results.AUC);
fprintf(fid,'\t\tAccuracy = %f\n', results.Accuracy);
fprintf(fid,'\t\tSensitivity or Recall = %f\n', results.Sensitivity);
fprintf(fid,'\t\tSpecificity = %f\n', results.Specificity);
fprintf(fid,'\t\tPrecision = %f\n', results.Precision);
fprintf(fid,'\t\tf1-score = %f\n', results.f1_score);
fprintf(fid,'\t\tMatthews correlation coefficient (MCC) = %f\n', results.MCC);
fprintf(fid,'\t\tJaccard index = %f\n', results.Jaccard_ind);
fprintf(fid,'\t\tCohen''s Kappa coefficient = %f\n', results.kappa);
fprintf(fid,'\n\t\tConfusion Matrix\n');
for i=1:size(results.ConfusionMatrix,1)
    for j=1:size(results.ConfusionMatrix,2)
        fprintf(fid,'\t\t\t%3d',results.ConfusionMatrix(i,j));
    end
    fprintf(fid,'\n\n');
end
fprintf(fid,'\tElapsed Time %f seconds\n', T);
fprintf(fid,'----------------------------------------------------------------------\n');
fclose(fid);