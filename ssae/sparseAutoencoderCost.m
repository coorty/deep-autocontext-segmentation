function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)
%% 注意:本文件所在路径: F:\SAE_脑白质分割_0805\USER_brainTissues_segmentation\Sparse_autoencoder
% 我这里决定把与稀疏约束有关的内容去除

% 输入：
% visibleSize:   输入层神经元数目 
% hiddenSize:    隐层神经元数目 
% lambda:        权重衰减参数

% 把稀疏约束去除
% sparsityParam: 稀疏约束目标激活值参数
% beta:          稀疏惩罚权重

% data:  训练数据，每一列是一个样本 

% 由于minFunc的计算要求，AE的权重以及偏置被弄成一个向量的形式
% AE所有的权重值
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);

b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

%% 梯度下降算法
[~, numSamples] = size(data);  % numSamples为样本的个数，n为样本的特征数

% 前向传播------计算各神经网络节点的线性组合值和active值

% 隐层 
z2 = bsxfun(@plus, W1 * data, b1);   % 我觉得这种方式更好（b1是个列向量）
a2 = sigmoid(z2);

% 输出层
z3 = bsxfun(@plus, W2 * a2, b2);
a3 = sigmoid(z3);

% cost的第一项： 计算预测产生的误差（均方误差）
Jcost = sum(sum((a3 - data).^2)) * 0.5 / numSamples;

% cost的第二项： 计算权值惩罚项
Jweight = 0.5 * (sum(sum(W1.^2)) + sum(sum(W2.^2)));

% cost = Jcost  +  lambda * Jweight;

% cost的第三项： 计算稀释性规则项(所有样本对某个隐层神经元激活平均)
rho = sum(a2,2) ./ numSamples;
Jsparse = sum(sparsityParam.*log(sparsityParam./rho)+ ...
        (1-sparsityParam).*log((1-sparsityParam)./(1-rho)));

% 损失函数的总表达式
cost = Jcost  +  lambda * Jweight  +  beta * Jsparse;

%% 反向算法求出每个节点的误差值
% 输出层units的残差
d3    = -(data-a3) .* sigmoidInv(z3);

%    计算W2grad  
W2grad = d3 * a2';
W2grad = W2grad ./ numSamples + lambda * W2;

%    计算b2grad 
b2grad = sum(d3,2);
b2grad = b2grad ./numSamples ;

% 加入了稀疏规则项，计算偏导的时候要引入此项
sterm = beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));   % 因为加入了稀疏规则项，所以
                                                                 % 计算偏导时需要引入该项
d2 = (W2'*d3 + repmat(sterm,1,numSamples)) .* sigmoidInv(z2); 
% d2 = (W2'*d3) .* sigmoidInv(z2); 

%  计算W1grad 
W1grad = d2*data';
W1grad = W1grad ./ numSamples + lambda * W1;

%  计算b1grad 
b1grad = sum(d2,2);
b1grad = b1grad ./ numSamples;   % 注意b的偏导是一个向量，所以这里应该把每一行的值累加起来

% 转变成一个向量有利于minFunc
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

% sigm 激活函数
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

% sigmoid函数的逆向求导函数
function sigmInv = sigmoidInv(x)
    sigmInv = sigmoid(x).*(1-sigmoid(x));
end