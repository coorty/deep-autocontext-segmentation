function ACMask = auto_context_mask(R)
%%% Generating mask of the auto-context features
% Input:
% ------
% R: radius intervals
%
% Output:
% ------
% ACMask: mask of the auto-context features
%

angle = 0:360/8:360;

batch = cell(numel(R), 1);
for j = 1:numel(R)
    batch{j} = zeros(R(end)*2+9, R(end)*2+9);
end

batch2 = cell(numel(R), 1);
for j = 1:numel(R)
    batch2{j} = zeros(R(end)*2+9, R(end)*2+9);
end

centerR = (size(batch{1}, 1) - 1) / 2;
centerC = (size(batch{1}, 2) - 1) / 2;
    
for num = 1:numel(R)
    for i = 1:numel(angle)-1
        xx = centerR + round(R(num) * cosd(angle(i)));
        yy = centerC + round(R(num) * sind(angle(i)));
        batch{num}(xx, yy) = 1;
    end
    
    [rows, cols] = find(batch{num} == 1);
    for j = 1:numel(rows)
        batch2{num}(rows(j)-1:rows(j)+1, cols(j)-1:cols(j)+1) = 1;
    end
end

finalBatch = false(R(end)*2+9, R(end)*2+9);
for k = 1:numel(R)
    finalBatch = finalBatch | logical(batch{k});
end

finalBatch(centerR-5:centerR+5, centerC-5:centerC+5) = 1;
ACMask = uint8(finalBatch);

end
