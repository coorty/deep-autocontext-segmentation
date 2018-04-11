function [ cost, grad ] = stackedAECost(theta, hiddenSize, numClasses, netconfig,lambda, traindata, labels)
%% 功能：stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for finetuning.
%  输入：                                         
%      theta: trained weights from the autoencoder(包含两个AE和一个softmax)
%      visibleSize: the number of input units
%      hiddenSize:  the number of hidden units *at the 2nd layer*
%      numClasses:  the number of categories
%      netconfig:   the network configuration of the stack
%      lambda:      the weight regularization penalty
%      traindata: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
%      labels: A vector containing labels, where labels(i) is the label for the  i-th training example
%  输出：
%      cost:
%      grad

%% Unroll softmaxTheta parameter

% 提取softmax的参数值
softmaxTheta = reshape(theta(1:hiddenSize * numClasses), numClasses, hiddenSize);

% 提取两层AE的参数值
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
% softmaxThetaGrad = zeros(size(softmaxTheta));

% 两个AE的权值和偏置
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

% cost = 0; % You need to compute this

% You might find these variables useful
M = size(traindata, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
% 计算SAE的代价函数和梯度向量
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% SAE的深度，这里为2
depth = numel(stack);

z  = cell(depth+1,1);
a  = cell(depth+1,1);

% a{1}存放了训练数据,z{1}没有用到
a{1} = traindata;

% 前向传播计算
for layer = 1:depth
    z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
    a{layer+1} = sigmoid(z{layer+1});
end

M = softmaxTheta * a{depth+1};
M = bsxfun(@minus, M, max(M));
p = bsxfun(@rdivide, exp(M), sum(exp(M)));

cost = -1/numClasses * groundTruth(:)' * log(p(:)) + lambda/2 * sum(softmaxTheta(:) .^ 2);

softmaxThetaGrad = -1/numClasses * (groundTruth - p) * a{depth+1}' + lambda * softmaxTheta;

d = cell(depth+1,1);

d{depth+1} = -(softmaxTheta' * (groundTruth - p)) .* a{depth+1} .* (1-a{depth+1});

for layer = (depth:-1:2)
  d{layer} = (stack{layer}.w' * d{layer+1}) .* a{layer} .* (1-a{layer});
end

for layer = (depth:-1:1)
  stackgrad{layer}.w = (1/numClasses) * d{layer+1} * a{layer}';
  stackgrad{layer}.b = (1/numClasses) * sum(d{layer+1}, 2);
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end

% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end