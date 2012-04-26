function seqrepeat = repeatability_evaluation( base_path, dataset_name, det_suffix, is_affine, x_val_file )

data_path = [base_path dataset_name '/'];
graph_path = [data_path 'results/'];

fprintf(1,'Data path: %s\n', data_path);
fprintf(1,'Graph path: %s\n', graph_path);


mkdir(data_path);
mkdir(graph_path);

mark=get_marks();

num_det = size(det_suffix,1);
[ x_label x_label_full x_unit x_vals ] = get_vals( data_path, x_val_file );
num_img = size( x_vals, 2 ) + 1;


setup_figure(1);
ylabel('repeatability')
xlabel(x_label_full);
set(gca,'YTick',[0.0 0.2 0.4 0.6 0.8 1.0]);
hold on;

setup_figure(2);
ylabel('nb of correspondences')
xlabel(x_label_full);
hold on;

setup_figure(3);
ylabel('matching score')
xlabel(x_label_full);
set(gca,'YTick',[0.0 0.2 0.4 0.6 0.8 1.0]);
hold on;

for d=1:num_det
    
    seqrepeat=[];
    seqcorresp=[];
    seqmatchscore=[];
    
    for i=2:num_img
        file1=sprintf('%skeypoints/img1.%s',data_path,char(det_suffix(d)));
        file2=sprintf('%skeypoints/img%d.%s',data_path,i,char(det_suffix(d)));
        Hom=sprintf('%sH1to%dp',data_path,i);
        imf1=sprintf('%simg1.ppm',data_path);
        imf2=sprintf('%simg%d.ppm',data_path,i);
        maskf=sprintf('%smask.pgm',data_path);
        
        [erro,repeat,corresp, match_score,matches, twi]=repeatability(file1,file2,Hom,imf1,imf2,maskf,1,not (is_affine{d}));
        seqrepeat=[seqrepeat repeat(4)];
        seqcorresp=[seqcorresp corresp(4)];
        seqmatchscore=[seqmatchscore match_score];
    end
    
    sfigure(1);  plot(x_vals,seqrepeat*0.01,mark{d},'LineWidth',4);
    sfigure(2);  plot(x_vals,seqcorresp,mark{d},'LineWidth',4);
    sfigure(3);  plot(x_vals,seqmatchscore*0.01,mark{d},'LineWidth',4);

end

for f=1:3
    sfigure(f);
    setup_axes( x_vals, num_img, [0 1] );
end

sfigure(2);
axis 'auto y'
pause(0.1);

print(sfigure(1),'-dpdf',sprintf('%srepeatability.pdf',graph_path))
print(sfigure(2),'-dpdf',sprintf('%snum_correspondences.pdf',graph_path))
print(sfigure(3),'-dpdf',sprintf('%smatching_score.pdf',graph_path))

figure(1)
legend(det_suffix,'Location','NorthOutside');
print(sfigure(1),'-dpdf',sprintf('%slegend.pdf',graph_path))

