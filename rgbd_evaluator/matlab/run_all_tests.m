
detectors = {'DAFT'};%;'DAFT Non-Affine';'SIFT'};
is_affine = {1,1,0,0,0};

%detectors = {'ORB';'SIFT'};
%is_affine = {0,0};

%detectors = {'SIFT'};
%is_affine = {0,0,0};

bagpath = '/work/gossow/bags/';

%complete_evaluation( [bagpath 'world_map/'], {'scale_2'}, detectors, 'scaling' );
%return;

complete_evaluation( [bagpath 'granada/'], {'rotate60_1'}, detectors, is_affine, 'rotation' );
return;

%complete_evaluation( [bagpath 'granada/'], {'viewpoint_1'}, detectors, is_affine, 'viewpoint angle' );

complete_evaluation( [bagpath 'tum_poster/'], {'viewpoint'}, detectors, is_affine, 'viewpoint angle' );
complete_evaluation( [bagpath 'tum_poster/'], {'scale'}, detectors, is_affine, 'scaling' );
complete_evaluation( [bagpath 'tum_poster/'], {'rotate'}, detectors, is_affine, 'rotation' );
return;

complete_evaluation( [bagpath 'granada/'], {'rotate40_1','rotate60_1'}, detectors, is_affine, 'rotation' );

complete_evaluation( [bagpath 'frosties/'], {'viewpoint'}, detectors, is_affine, 'viewpoint angle' );


complete_evaluation( [bagpath 'world_map/'], {'viewpoint0_1', 'viewpoint22_1', 'viewpoint45_1'}, detectors, is_affine, 'viewpoint angle' );
complete_evaluation( [bagpath 'world_map/'], {'scale_1','scale_2'}, detectors, is_affine, 'scaling' );
complete_evaluation( [bagpath 'world_map/'], {'rotate0_1', 'rotate22_1', 'rotate45_1'}, detectors, is_affine, 'rotation' );

%complete_evaluation( [bagpath 'ecai_poster/'], {'rotate_1'}, detectors, 'rotation' );
%complete_evaluation( [bagpath 'ecai_poster/'], {'scale_1'}, detectors, 'scaling' );
%complete_evaluation( [bagpath 'ecai_poster/'], {'viewpoint_1'}, detectors, 'viewpoint angle' );


%---------

%complete_evaluation( [bagpath 'world_map/'], {'rotate45_1'}, detectors, 'rotation' );
%complete_evaluation( [bagpath 'world_map/'], {'viewpoint45_1'}, detectors, 'viewpoint angle' );
