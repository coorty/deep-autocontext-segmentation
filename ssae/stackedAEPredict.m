function [predvalue, pred] = stackedAEPredict(theta,hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1, 1);

numSamples = size(data,2);
pred = zeros(1,numSamples);

if numSamples > 120000
    
    for i=1:floor(numSamples/100000)
        a{1} = data(:,(i-1)*100000+1:i*100000);
        for layer = (1:depth)
            z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
            a{layer+1} = sigmoid(z{layer+1});
        end
        [predvalue2, pred2] = max(exp(softmaxTheta * a{depth+1}));
        pred((i-1)*100000+1:i*100000) = pred2;
        predvalue((i-1)*100000+1:i*100000) = predvalue2;
        
    end
    
    if mod(numSamples,100000) ~= 0
       a{1} = data(:,i*100000+1:end);
       for layer = (1:depth)
          z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
          a{layer+1} = sigmoid(z{layer+1});
       end
       [predvalue2, pred2] = max(exp(softmaxTheta * a{depth+1}));
       
       pred(i*100000+1:end) = pred2;
       predvalue((i-1)*100000+1:i*100000) = predvalue2;
       
    end
    
else
    a{1} = data;
    for layer = (1:depth)
      z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
      a{layer+1} = sigmoid(z{layer+1});
    end   
    
    [predvalue, pred] = max(exp(softmaxTheta * a{depth+1}));     % 概率最大的那个
    
end
% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end