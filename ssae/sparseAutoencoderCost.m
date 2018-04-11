function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)
%% ע��:���ļ�����·��: F:\SAE_�԰��ʷָ�_0805\USER_brainTissues_segmentation\Sparse_autoencoder
% �������������ϡ��Լ���йص�����ȥ��

% ���룺
% visibleSize:   �������Ԫ��Ŀ 
% hiddenSize:    ������Ԫ��Ŀ 
% lambda:        Ȩ��˥������

% ��ϡ��Լ��ȥ��
% sparsityParam: ϡ��Լ��Ŀ�꼤��ֵ����
% beta:          ϡ��ͷ�Ȩ��

% data:  ѵ�����ݣ�ÿһ����һ������ 

% ����minFunc�ļ���Ҫ��AE��Ȩ���Լ�ƫ�ñ�Ū��һ����������ʽ
% AE���е�Ȩ��ֵ
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);

b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

%% �ݶ��½��㷨
[~, numSamples] = size(data);  % numSamplesΪ�����ĸ�����nΪ������������

% ǰ�򴫲�------�����������ڵ���������ֵ��activeֵ

% ���� 
z2 = bsxfun(@plus, W1 * data, b1);   % �Ҿ������ַ�ʽ���ã�b1�Ǹ���������
a2 = sigmoid(z2);

% �����
z3 = bsxfun(@plus, W2 * a2, b2);
a3 = sigmoid(z3);

% cost�ĵ�һ� ����Ԥ���������������
Jcost = sum(sum((a3 - data).^2)) * 0.5 / numSamples;

% cost�ĵڶ�� ����Ȩֵ�ͷ���
Jweight = 0.5 * (sum(sum(W1.^2)) + sum(sum(W2.^2)));

% cost = Jcost  +  lambda * Jweight;

% cost�ĵ���� ����ϡ���Թ�����(����������ĳ��������Ԫ����ƽ��)
rho = sum(a2,2) ./ numSamples;
Jsparse = sum(sparsityParam.*log(sparsityParam./rho)+ ...
        (1-sparsityParam).*log((1-sparsityParam)./(1-rho)));

% ��ʧ�������ܱ��ʽ
cost = Jcost  +  lambda * Jweight  +  beta * Jsparse;

%% �����㷨���ÿ���ڵ�����ֵ
% �����units�Ĳв�
d3    = -(data-a3) .* sigmoidInv(z3);

%    ����W2grad  
W2grad = d3 * a2';
W2grad = W2grad ./ numSamples + lambda * W2;

%    ����b2grad 
b2grad = sum(d3,2);
b2grad = b2grad ./numSamples ;

% ������ϡ����������ƫ����ʱ��Ҫ�������
sterm = beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));   % ��Ϊ������ϡ����������
                                                                 % ����ƫ��ʱ��Ҫ�������
d2 = (W2'*d3 + repmat(sterm,1,numSamples)) .* sigmoidInv(z2); 
% d2 = (W2'*d3) .* sigmoidInv(z2); 

%  ����W1grad 
W1grad = d2*data';
W1grad = W1grad ./ numSamples + lambda * W1;

%  ����b1grad 
b1grad = sum(d2,2);
b1grad = b1grad ./ numSamples;   % ע��b��ƫ����һ����������������Ӧ�ð�ÿһ�е�ֵ�ۼ�����

% ת���һ������������minFunc
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

% sigm �����
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

% sigmoid�����������󵼺���
function sigmInv = sigmoidInv(x)
    sigmInv = sigmoid(x).*(1-sigmoid(x));
end