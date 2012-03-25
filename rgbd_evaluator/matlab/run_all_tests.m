
detectors = {'DAFT Affine';'DAFT-Fast Affine';'SIFT';'SURF'};
%detectors = {'DAFT Affine'};

bagpath = '/work/gossow/bags/';

%complete_evaluation( bagpath ,'tum_poster/viewpoint', detectors, 'viewpoint angle' );
%complete_evaluation( bagpath ,'tum_poster/scale', detectors, 'viewpoint angle' );
%complete_evaluation( bagpath ,'tum_poster/rotate', detectors, 'viewpoint angle' );

complete_evaluation( bagpath ,'world_map/viewpoint0_1', detectors, 'viewpoint angle' );
