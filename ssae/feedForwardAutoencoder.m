function sae1Features = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, inputSize, trainData)
%%
W1 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
b1 = sae1OptTheta(2 * hiddenSizeL1 * inputSize + 1 : 2 * hiddenSizeL1 * inputSize + hiddenSizeL1);
z1 = bsxfun(@plus, W1 * trainData, b1);
sae1Features = sigmoid(z1);
end

% sigm ¼¤»îº¯Êý
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end








