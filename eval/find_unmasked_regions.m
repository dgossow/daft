function [ ind ] = find_unmasked_regions( feat, im, mask, show_keypoints )

h=size(im,1);
w=size(im,2);

mask2 = mask;

% check bounding boxes against image borders
ind1=find((feat(:,1)+feat(:,8))<w & (feat(:,1)-feat(:,8))>0 & (feat(:,2)+feat(:,9))<h & (feat(:,2)-feat(:,9))>0);

ind = [];
for i=1:size(feat,1)
    f = ind1(i);
    
    %fprintf(1,'%d \n', mask( int32(feat(f,2)), int32(feat(f,1)) ) );
    
    if mask( int32(feat(f,2)), int32(feat(f,1)) ) > 128
        ind = [ ind; f ];
        mask2( int32(feat(f,2)), int32(feat(f,1)) ) = 0;
    else
        %fprintf(1,'feature at %d %d is outside the mask.\n', feat(f,1), feat(f,2));
        mask2( int32(feat(f,2)), int32(feat(f,1)) ) = 255;
    end       
end

fprintf(1,'%d features are outside the mask.\n', (size(ind1,1)-size(ind,1)) );
fprintf(1,'%d features remaining.\n', size(ind,1) );


if show_keypoints==1
sfigure(7);
imshow(mask2);
drawnow;
end

end

