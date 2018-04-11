function [r,c] = get_contour_center(contourImg)
%%% Get the center coordinate of the contour image
% Input:
% ------
% contourImg: 
%
% Output:
% -------
% r: row coordinate
% c: col coordinate
%

lab = zeros(size(contourImg));

lab(contourImg == max(contourImg(:))) = 1;
contourImg = lab; 
clear lab;

ContourCenter = zeros(size(contourImg, 3), 2);

for i = 1:size(contourImg, 3)
    tmp = regionprops(squeeze(contourImg(:,:,i)), 'Centroid');
    ContourCenter(i, :) = floor(tmp.Centroid);
end

r = floor(ContourCenter(:,2));
c = floor(ContourCenter(:,1));
