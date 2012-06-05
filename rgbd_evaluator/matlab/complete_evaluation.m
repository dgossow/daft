function complete_evaluation( base_path, summary_name, dataset_names, det_suffix, is_affine, x_val_file )

data_path = [base_path summary_name '/'];
graph_path = data_path;

fprintf(1,'Data path: %s\n', data_path);
fprintf(1,'Graph path: %s\n', graph_path);

mkdir(data_path);
mkdir(graph_path);

matrepeat = [];
matcorresp = [];
matf1scores = [];
matxvals = [];

for dataset_name = dataset_names
    
    data_path = strcat(base_path, char(dataset_name), '/')
    
    [ x_label x_label_full x_unit x_vals ] = get_vals( data_path, x_val_file );

    [ r c ] = repeatability_evaluation( base_path, char(dataset_name), det_suffix, is_affine, x_val_file );
    matrepeat = [ matrepeat r ];
    matcorresp = [ matcorresp c ];
    matxvals = [ matxvals x_vals ];
    
    f1 = descperf_evaluation( base_path, char(dataset_name), det_suffix, is_affine, x_val_file );
    matf1scores = [ matf1scores f1 ];
    
end

% plot repeatability

setup_figure(1);
xlabel(x_label_full);
ylabel('repeatability')
set(gca,'YTick',[0.0 0.2 0.4 0.6 0.8 1.0]);
hold on;

setup_figure(2);
xlabel(x_label_full);
ylabel('nb of correspondences')
hold on;

setup_figure(3);
xlabel(x_label_full);
ylabel('max f1 score');
%set(gca,'YTick',[0.0 0.2 0.4 0.6 0.8 1.0]);
hold on;

mark=get_marks();

num_det = size(det_suffix,1);

for d=1:num_det
    
    sfigure(1);
    smooth_plot(5,matxvals,matrepeat(d,:)*0.01,mark{d},'LineWidth',4);
    
    sfigure(2);
    smooth_plot(5,matxvals,matcorresp(d,:),mark{d},'LineWidth',4);

    sfigure(3);
    smooth_plot(3,matxvals,matf1scores(d,:),mark{d},'LineWidth',4);
end

sfigure(2);
axis 'auto y'
pause(0.1);

print(sfigure(1),'-dpdf',sprintf('%srepeatability.pdf',graph_path))
print(sfigure(2),'-dpdf',sprintf('%snum_correspondences.pdf',graph_path))
print(sfigure(3),'-dpdf',sprintf('%sf1score.pdf',graph_path))

figure(1)
legend(det_suffix,'Location','NorthOutside');
print(sfigure(1),'-dpdf',sprintf('%slegend.pdf',graph_path))


