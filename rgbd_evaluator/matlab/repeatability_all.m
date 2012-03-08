
detectors = {'DAFT Affine';'DAFT-Fast Affine';'SIFT';'SURF'};
%detectors = {'DAFT-Fast Affine';'DAFT Affine';'DAFT';'DAFT-Fast'};

bagpath = '/work/gossow/bags/eval/';

repeatability_evaluation( bagpath ,'poster_viewpoint', detectors, 'viewpoint angle', 8 );
repeatability_evaluation( bagpath ,'poster_scale', detectors, 'scaling', 7);
repeatability_evaluation( bagpath ,'poster_rotate', detectors, 'rotation', 8 );
