function complete_evaluation( base_path, summary_name, dataset_names, det_suffix, is_affine, x_val_file )

data_path = [base_path summary_name '/'];
graph_path = data_path;

fprintf(1,'Data path: %s\n', data_path);
fprintf(1,'Graph path: %s\n', graph_path);

mkdir(data_path);
mkdir(graph_path);

matrepeat = [];
matcorresp = [];
matxvals = [];

for dataset_name = dataset_names
    
    data_path = strcat(base_path, char(dataset_name), '/')
    
    [ x_label x_label_full x_unit x_vals ] = get_vals( data_path, x_val_file );

    [ r c ] = repeatability_evaluation( base_path, char(dataset_name), det_suffix, is_affine, x_val_file );
    matrepeat = [ matrepeat r ];
    matcorresp = [ matcorresp c ];
    matxvals = [ matxvals x_vals ];
    pause(3);
    %descperf_evaluation( base_path, char(dataset_name), det_suffix, is_affine, x_val_file );
    
end

% plot repeatability

setup_figure(1);
ylabel('repeatability')
xlabel(x_label_full);
set(gca,'YTick',[0.0 0.2 0.4 0.6 0.8 1.0]);
hold on;

setup_figure(2);
ylabel('nb of correspondences')
xlabel(x_label_full);
hold on;

mark=get_marks();

num_det = size(det_suffix,1);

for d=1:num_det
    
    sfigure(1);
    smooth_plot(5,matxvals,matrepeat(d,:)*0.01,mark{d},'LineWidth',4);
    
    sfigure(2);
    smooth_plot(5,matxvals,matcorresp(d,:),mark{d},'LineWidth',4);
    
end

for f=1:2
    sfigure(f);
    setup_axes( x_vals, [0 1] );
end

sfigure(2);
axis 'auto y'
pause(0.1);

print(sfigure(1),'-dpdf',sprintf('%srepeatability.pdf',graph_path))
print(sfigure(2),'-dpdf',sprintf('%snum_correspondences.pdf',graph_path))

figure(1)
legend(det_suffix,'Location','NorthOutside');
print(sfigure(1),'-dpdf',sprintf('%slegend.pdf',graph_path))


