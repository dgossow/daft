function complete_evaluation( base_path, dataset_names, det_suffix, is_affine, x_val_file )

for dataset_name = dataset_names
    repeatability_evaluation( base_path, char(dataset_name), det_suffix, is_affine, x_val_file )
    descperf_evaluation( base_path, char(dataset_name), det_suffix, is_affine, x_val_file );
end
