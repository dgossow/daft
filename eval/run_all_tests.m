
detectors = {'DAFT';'SIFT'};
detectors = {'DAFT';'DAFT Gauss3D'};
%detectors = {'DAFT';'DAFT Non-Affine'};
is_affine = {1,1,1,1,0};

%detectors = {'ORB';'SIFT'};
%is_affine = {0,0};

%detectors = {'SIFT'};
%is_affine = {0,0,0};

bagpath = '/work/gossow/bags/';

%complete_evaluation( bagpath, 'eval/test', {'granada/scale'}, detectors, is_affine, 'scaling' );
%return;

%complete_evaluation( bagpath, 'granada', {'granada/rotate60_1'}, detectors, is_affine, 'rotation' );
%complete_evaluation( bagpath, 'granada', {'granada/rotate40_1'}, detectors, is_affine, 'rotation' );

%{'poster/vprotate35','poster/vprotate45','poster/vprotate55'}

complete_evaluation( bagpath, 'eval/test', {'poster/camrotate60'}, detectors, is_affine, 'camera rotation' );
return;

complete_evaluation( bagpath, 'eval/vprotation', {'poster/vprotate35','poster/vprotate40','poster/vprotate45','poster/vprotate55'}, detectors, is_affine, 'rotation' );

complete_evaluation( bagpath, 'eval/scale', {'poster/scale'}, detectors, is_affine, 'scaling' );

viewpoint_datasets={'poster/viewpoint','poster/viewpoint22','poster/viewpoint45'};
complete_evaluation( bagpath, 'eval/viewpoint', viewpoint_datasets, detectors, is_affine, 'viewpoint angle' );

complete_evaluation( bagpath, 'eval/cam_rotate', {'poster/camrotate','poster/camrotate45','poster/camrotate60'}, detectors, is_affine, 'camera rotation' );

return;

rotate_datasets = {
    'world_map/rotate0_1', ...
    'world_map/rotate22_1',  ...
    'world_map/rotate45_1', ...
    'tum_poster/rotate' };
complete_evaluation( bagpath, 'eval/rotate', rotate_datasets, detectors, is_affine, 'rotation' );

scale_datasets = {
    'world_map/scale_1', ...
    'world_map/scale_2', ...
    'tum_poster/scale' ...
    }
complete_evaluation( bagpath, 'eval/scale', scale_datasets, detectors, is_affine, 'scaling' );

return;

%'honeyloops/viewpoint', ...
viewpoint_datasets={ ...
    'tum_poster/viewpoint', ...
    'frosties/viewpoint', ...
    'world_map/viewpoint0_1', ...
    'world_map/viewpoint22_1', ...
    'world_map/viewpoint45_1' ...
    };
%viewpoint_datasets={ 'honeyloops/viewpoint', 'frosties/viewpoint' };

complete_evaluation( bagpath, 'eval/viewpoint', viewpoint_datasets, detectors, is_affine, 'viewpoint angle' );

scale_datasets = {
    'world_map/scale_1', ...
    'world_map/scale_2', ...
    'tum_poster/scale' ...
    }
complete_evaluation( bagpath, 'eval/scale', scale_datasets, detectors, is_affine, 'scaling' );

%complete_evaluation( [bagpath 'ecai_poster/'], {'rotate_1'}, detectors, 'rotation' );
%complete_evaluation( [bagpath 'ecai_poster/'], {'scale_1'}, detectors, 'scaling' );
%complete_evaluation( [bagpath 'ecai_poster/'], {'viewpoint_1'}, detectors, 'viewpoint angle' );


%---------

%complete_evaluation( [bagpath 'world_map/'], {'rotate45_1'}, detectors, 'rotation' );
%complete_evaluation( [bagpath 'world_map/'], {'viewpoint45_1'}, detectors, 'viewpoint angle' );
