
detectors = {'DAFT Affine';'DAFT-Fast Affine';'SIFT';'SURF'};
%detectors = {'DAFT Affine'};

bagpath = '/work/gossow/bags/';

%complete_evaluation( [bagpath 'world_map/'], {'rotate0_1', 'rotate45_1', 'rotate22_1'}, detectors, 'rotation' );
%complete_evaluation( [bagpath 'world_map/'], {'scale_1', 'scale_2'}, detectors, 'scaling' );
%complete_evaluation( [bagpath 'world_map/'], {'viewpoint0_1', 'viewpoint22_1', 'viewpoint45_1'}, detectors, 'viewpoint angle' );

complete_evaluation( [bagpath 'tum_poster/'], {'viewpoint'}, detectors, 'viewpoint angle' );
%complete_evaluation( [bagpath 'tum_poster/'], {'scale'}, detectors, 'scaling' );
complete_evaluation( [bagpath 'tum_poster/'], {'rotate'}, detectors, 'rotation' );

complete_evaluation( [bagpath 'ecai_poster/'], {'rotate_1'}, detectors, 'rotation' );
complete_evaluation( [bagpath 'ecai_poster/'], {'scale_1'}, detectors, 'scaling' );
complete_evaluation( [bagpath 'ecai_poster/'], {'viewpoint_1'}, detectors, 'viewpoint angle' );
