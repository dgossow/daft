function [ axis_label, axis_label_full, axis_unit, axis_vals ] = get_vals( data_path, val_file )

axis_vals = load( sprintf( '%s%s', data_path, val_file ) );
axis_unit = '';
axis_label = '';

if ( strcmp(val_file,'viewpoint angle') )
    axis_label = 'viewpoint angle';
    axis_unit = '\circ';
elseif ( strcmp(val_file,'scaling') )
    axis_label = 'scaling';
    axis_unit = '';
elseif ( strcmp(val_file,'rotation') )
    axis_label = 'rotation';
    axis_unit = '\circ';
end

if ( strcmp(axis_unit,'') == 0 )
    axis_label_full = [axis_label ' (' axis_unit ')'];
else
    axis_label_full = axis_label;
end

end

