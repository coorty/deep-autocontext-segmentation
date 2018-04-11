function [ layer3_out ] = extract_deep_learned_feature(deepnet, samples)
%%% Extracting deep learned features from the deep net
% Input:
% ------
% deepnet: A trained deep network
% samples: samples that the deepnet performed on
%
% Return:
% -------
% layer3_out: deep feature that the deepnet learned on samples
%

% This is a forward process
layer1_out = logsig(bsxfun(@plus, deepnet.IW{1} * mapminmax(samples,0,1), deepnet.b{1}));
layer2_out = logsig(bsxfun(@plus, deepnet.LW{2,1} * layer1_out, deepnet.b{2}));
layer3_out = logsig(bsxfun(@plus, deepnet.LW{3,2} * layer2_out, deepnet.b{3}));

% softmax_out = exp(bsxfun(@plus,deepnet.LW{4,3}*layer3_out,deepnet.b{4}));
% softmax_out = bsxfun(@rdivide,softmax_out,sum(softmax_out));

end

