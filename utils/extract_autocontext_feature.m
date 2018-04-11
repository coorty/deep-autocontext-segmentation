function [imageACMPatches] = extract_autocontext_feature(map, mask)
%%% Extract auto-context features from the posterior probability maps
% Input:
% ------
% map: probability maps
% mask: ACM mask
%
% Output:
% -------
% imageACMPatches: auto-context features
%

[rows, cols] = size(map);
ACMDim       = sum(mask(:));   % Dimension of the auto-context feature
halfMaskSize = (size(mask, 1)-1)/2;

% Initialization
imageACMPatches = zeros(ACMDim, numel(maps(:)));

% Pad the boundary with 0
maps = padarray(maps,[halfMaskSize halfMaskSize], 0);

samplesNum = 0;
for cc = halfMaskSize+1:cols-halfMaskSize
    for rr = halfMaskSize+1:rows-halfMaskSize
        samplesNum = samplesNum + 1;
        currPatch  = map(rr-halfMaskSize:rr+halfMaskSize, cc-halfMaskSize:cc+halfMaskSize);
        currACM    = currPatch(find(mask > 0));
        imageACMPatches(:,samplesNum) = currACM(:);
    end
end

end