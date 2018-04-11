function [imagePatches, imageLabels] = extract_image_patches(image, label, halfPatchSize)
%%% Extract whole patches from image
% Input:
% ------
% image: gray image
% label: label of the image
% halfPatchSize: half of the patch's size
%
% Return:
% -------
% all patches and label of the image
%

if size(image, 3) == 3
	image = rgb2gray(image)
end

[rows, cols] = size(image);
patchSize = halfPatchSize * 2 + 1;
sampleNum = rows * cols;  

% Pad the boundary with 0
image = padarray(image,[halfPatchSize halfPatchSize], 0);
imagePatches = zeros(patchSize*patchSize, sampleNum);

samplesNum = 0;
for cc = halfPatchSize+1:cols-halfPatchSize
    for rr = halfPatchSize+1:rows-halfPatchSize
        samplesNum = samplesNum + 1;
        currPatch = image(rr-halfPatchSize:rr+halfPatchSize, cc-halfPatchSize:cc+halfPatchSize);
        imagePatches(:,samplesNum) = currPatch(:);
    end
end

if nargout ~= 1
    imageLabels = label(:)';
end

end 