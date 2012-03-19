
%detectors = {'SIFT';'DAFT-Fast Affine'};
detectors = {'DAFT Affine';'DAFT-Fast Affine';'SIFT';'SURF'};

bagpath = '/work/gossow/bags/eval/';

descperf_evaluation( bagpath ,'poster_scale', detectors, 'scaling', 7);
descperf_evaluation( bagpath ,'poster_viewpoint', detectors, 'viewpoint angle', 8 );
descperf_evaluation( bagpath ,'poster_rotate', detectors, 'rotation', 8 );
