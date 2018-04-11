function [Features,stack]  = ssaeTrain(TrainData, cascadeNum)
%% ѵ��SSAE
% ʹ�ýṹ��
AE_options = struct; 
AE_options.Method  = 'lbfgs';   % �Ż�����
AE_options.maxIter = 200;       % �������ܴ���
AE_options.display = 'off';     % ��ʾ�м�����Ĺ���

global net

net.AE_options = AE_options;

stack = cell(net.ssaeDepth, 1);
Features = TrainData;
for i = 1:net.ssaeDepth
    disp(['    We are under pretraining the ', num2str(i), '/', num2str(net.ssaeDepth), ' AE.']),
    disp(['���ڵ�ʱ���ǣ� ', datestr(now,'mm-dd_HH.MM')])
    
    sae_Theta = initializeParameters(net.layersUnits{cascadeNum}(i+1), net.layersUnits{cascadeNum}(i));  % AE������ʼ��
      
    [saeOptTheta, cost] =  minFunc(@(p)sparseAutoencoderCost(p,...
        net.layersUnits{cascadeNum}(i), net.layersUnits{cascadeNum}(i+1), net.AElambda, net.sparsityParam, net.beta, Features), sae_Theta, AE_options); 
    disp(['The ', num2str(i), '/', num2str(net.ssaeDepth), ' pretrain AE cost is ', num2str(cost)]), 
    
    % ��ȡ��������
    Features = feedForwardAutoencoder(saeOptTheta, net.layersUnits{cascadeNum}(i+1), net.layersUnits{cascadeNum}(i), Features); 
    
    stack{i}.w = reshape(saeOptTheta(1:net.layersUnits{cascadeNum}(i+1) * net.layersUnits{cascadeNum}(i)), net.layersUnits{cascadeNum}(i+1), net.layersUnits{cascadeNum}(i));
    stack{i}.b = saeOptTheta(2*net.layersUnits{cascadeNum}(i+1)*net.layersUnits{cascadeNum}(i)+1:2*net.layersUnits{cascadeNum}(i+1)*net.layersUnits{cascadeNum}(i)+net.layersUnits{cascadeNum}(i+1));   
    
%     display_network(stack{i}.w', 12);
%     print('-djpeg', ['./result/W', num2str(i),'.jpg']); close
end















