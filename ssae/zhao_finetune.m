function [stackedAEOptTheta,netconfig] = zhao_finetune(TrainData, TrainLabels, stack, softmaxModel, cascadeNum)
%%
global net;
% 为深度模型初始化参数
[stackparams, netconfig] = stack2params(stack);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

% stackedAETheta是个向量，为整个网络的参数，包括分类器那部分，且分类器那部分的参数放前面
stackedAETheta = [saeSoftmaxOptTheta; stackparams];

disp('正在微调网络......')    
[stackedAEOptTheta, ~] =  minFunc(@(p)stackedAECost(p, net.layersUnits{cascadeNum}(end),...
                             net.numClasses, netconfig, net.AElambda, TrainData, TrainLabels),...
                            stackedAETheta, net.AE_options); 
                       
                        
                        
                        
                        
                        
                        
                        
                        
                        