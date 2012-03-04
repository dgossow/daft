
%detectors = {'DAFT affine';'DAFT';'SURF'};
detectors = {'DAFT'};% {'DAFT';'SIFT';'SURF'};
detectors = {'DAFT affine';'DAFT';'SIFT';'SURF'};
bagpath = '/work/gossow/bags/eval/';

repeatability_evaluation( bagpath ,'hallway_viewpoint', detectors, 'viewpoint angle', 6 )
%repeatability_evaluation( bagpath ,'poster_viewpoint', detectors, 'viewpoint angle', 8 )
%repeatability_evaluation( bagpath ,'poster_scale', detectors, 'scaling', 7 )
%repeatability_evaluation( bagpath ,'poster_rotate', detectors, 'rotation', 8 )
