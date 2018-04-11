function [ eva ] = evalute_segment_performance(gtLab, predLab)
%%% Using several metrics to evaluate the segmentation result.
% Input:
% ------
% gtLab: ground truth label
% predLab: the predicted label
%
% Output:
% -------
% eva: contain several evaluation metrics
%

% Using `precision`, `recall`, `DSC`, `Jaccard`, `Accuracy`
gtLab   = gtLab(:);
predLab = predLab(:);

M  = numel(gtLab);         % Number of elements

TP = sum(gtLab & predLab);
TN = numel(find((gtLab - predLab)==1));
FP = sum(predLab - gtLab & predLab);
FN = sum(xor(gtLab, predLab)) - FP;
P = TP + FN;
N = FP + TN;

Precision = TP / (TP + FP); 
Recall    = TP / (TP + FN); 

DSC = 2*(Precision * Recall)/(Precision+Recall); 
Jaccard = sum(~xor(gtLab, predLab)) / (M*2-sum(~xor(gtLab, predLab)));     
Accuracy = (TP+TN)/(P+N);

eva.Precision = Precision;
eva.Recall = Recall;
eva.DSC = DSC;
eva.Jaccard = Jaccard;
eva.Accuracy = Accuracy;
end

