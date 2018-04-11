% Date: 2016.05.13 09:35am Friday
% Author: zhaopace@foxmail.com
% Version: 1.0
% Desc: 
% 	Deep auto-context model used for object segmentation (applied in Weizmann Horses Dataset) 
% https://avaminzhang.wordpress.com/2012/12/07/%E3%80%90dataset%E3%80%91weizmann-horses/
%
% Revised Records:
%

% Clear workspace
clc; clear; close all;

% Dependency setting
dirBase = './';
addpath(fullfile(dirBase, 'utils'));
addpath(fullfile(dirBase, 'softmax'));
addpath(fullfile(dirBase, 'ssae'));

dirSaveParams = fullfile(dirBase, 'model');
dirData       = fullfile(dirBase, 'data');
dirResults    = fullfile(dirBase, 'results');

% Parameters setting
halfPatchSize = (19-1)/2; 
levelNum      = 7;
R = [3 5 7 9 11 14 17 20 23];   % Auto-context feature parameters

% Variables initialization
imgProbMaps = cell(levelNum+1,1);
imgEva      = cell(levelNum+1,1);
allSamples  = cell(levelNum+1,1);

% Load the trained models
load(fullfile(dirSaveParams, 'deepnet'));
load(fullfile(dirSaveParams, 'softmaxModel'));

% Name of the test image
imgName    = 'horse012.jpg'
imgLabName = 'horse012.jpg'

img = imread(fullfile(dirData, 'images', imgName));
lab = imread(fullfile(dirData, 'labels', imgLabName));
[rows,cols] = size(img);

% Extract all patches from image
[imgPatches, imgLabels] = extract_image_patches(img, lab, halfPatchSize);

% Extract deep feature representation from `imgPatches`
imgDeepFeature = extract_deep_learned_feature(deepnet, imgPatches);

% Extract posterior probability maps of the first level
probValue = deepnet(imgPatches);
imgProbMaps{1} = reshape(probValue(2,:),rows,cols);

% Extract Auto-context features from the posterior probability maps
imgACMPatches = extract_autocontext_feature(imgProbMaps{1}, auto_context_mask(R));

% Evaluate the segmentation performance of the first level
eva = evalute_segment_performance(lab, imgProbMaps{1}>=0.5);
disp(['level 1: Precision: ',num2str(eva.Precision),' | Recall: ',num2str(eva.Recall),...
    ' | DSC: ',num2str(eva.DSC),' | Jaccard: ',num2str(eva.Jaccard),'| Accuracy: ',num2str(eva.Accuracy)])
	
imgEva{1}     = eva;
allSamples{2} = [imgDeepFeature; imgACMPatches];	
	
% Loop	
for level = 2:levelNum
	% Execute classifying & get the new probability maps
    softnet   = softmaxModel{level};
    probValue = softnet(allSamples{level});
    imgProbMaps{level} = reshape(probValue(2,:), rows, cols);
    
	% Evaluation  
    eva = evalute_segment_performance(lab, imgProbMaps{level} >= 0.5);
    disp(['level ',num2str(level),': Precision: ',num2str(eva.Precision),...
        ' | Recall: ',num2str(eva.Recall),' | DSC: ',num2str(eva.DSC),...
        ' | Jaccard: ',num2str(eva.Jaccard),' | Accuracy: ',num2str(eva.Accuracy)])
		
    % Extract Auto-context features from the posterior probability maps
    imgACMPatches = extract_autocontext_feature(testimgProbMaps{level}, Auto_context_mask(R));
	
	imgEva{level} = eva;
    allSamples{level+1} = [imgDeepFec; imgACMPatches];	
end
	
% Save results
for level = 1:2:levelNum
	imwrite(mat2gray(testimgProbMaps{level}),fullfile(dirResults,[imgName(1:8),'_level',num2str(level),'.jpg']))
end

%save(fullfile(dirResults,'testimgProbMaps.mat'),'testimgProbMaps');
%save(fullfile(dirResults,'testimgEva.mat'),'testimgEva');
