function [stackedAEOptTheta,netconfig] = zhao_finetune(TrainData, TrainLabels, stack, softmaxModel, cascadeNum)
%%
global net;
% Ϊ���ģ�ͳ�ʼ������
[stackparams, netconfig] = stack2params(stack);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

% stackedAETheta�Ǹ�������Ϊ��������Ĳ����������������ǲ��֣��ҷ������ǲ��ֵĲ�����ǰ��
stackedAETheta = [saeSoftmaxOptTheta; stackparams];

disp('����΢������......')    
[stackedAEOptTheta, ~] =  minFunc(@(p)stackedAECost(p, net.layersUnits{cascadeNum}(end),...
                             net.numClasses, netconfig, net.AElambda, TrainData, TrainLabels),...
                            stackedAETheta, net.AE_options); 
                       
                        
                        
                        
                        
                        
                        
                        
                        
                        