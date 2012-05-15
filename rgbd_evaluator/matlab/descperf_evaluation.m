function [precision, recall] = descperf_evaluation( base_path, dataset_name, det_suffix, is_affine, x_val_file )

data_path = [base_path, dataset_name, '/'];
graph_path = [data_path, 'results/'];
mkdir(graph_path)

fprintf(1,'It may take a while, i.e. 30min \n');

num_det = size(det_suffix,1);

mark=get_marks();

%mark={'-rs';'--bp';'-kx';'--kv';':r+';'-.bp';'--b>'};

[ x_label x_label_full x_unit x_vals ] = get_vals( data_path, x_val_file );
num_img = size( x_vals, 2 ) + 1;

f1scores = zeros( num_det, num_img-1 );

for i=2:num_img

    setup_figure(1);
    set(gca,'XTick',[0.0 0.25 0.5 0.75 1.0]);
    set(gca,'YTick',[0.0 0.2 0.4 0.6 0.8 1.0]);
    grid on;
    ylabel('recall');
    xlabel('precision');
    title( sprintf('%s = %1.1f%s', x_label, x_vals(i-1), x_unit ) );
    hold on;
    
    for d=1:num_det
    
        file1=sprintf('%skeypoints/img1.%s',data_path,char(det_suffix(d)));
        file2=sprintf('%skeypoints/img%d.%s',data_path,i,char(det_suffix(d)));
        Hom=sprintf('%sH1to%dp',data_path,i);
        imf1=sprintf('%simg1.ppm',data_path);
        imf2=sprintf('%simg%d.ppm',data_path,i);

        [erro,repeat,corresp, match_score,matches, twi]=repeatability(file1,file2,Hom,imf1,imf2,'',0,not (is_affine{d}));

        [correct_match_nn,total_match_nn,correct_match_sim,total_match_sim,correct_match_rn,total_match_rn] = descperf(file1,file2,not (is_affine{d}),Hom,imf1,imf2,corresp(4),twi);

        recall=correct_match_rn./corresp(4);
        %precision=(total_match_rn-correct_match_rn)./total_match_rn
        precision=correct_match_rn./total_match_rn;

        plot( precision, recall, mark{d},'LineWidth',4 );

        f1score = max ( (2* precision .* recall) ./ (precision + recall) )
        f1scores( d, i-1 ) = f1score;
        
    end
    
    axis([0 1 0 1]);
    figname = sprintf('%sdescperf_%i.pdf',graph_path, i-1 );
    print(sfigure(1),'-dpdf',figname);
end

setup_figure(1);
sfigure(1);
setup_axes( x_vals, num_img, [0 1] );
%axis([0 1 0 1]);
xlabel(x_label_full);
ylabel('max f1 score');
set(gca,'YTick',[0.0 0.2 0.4 0.6 0.8 1.0]);
grid on;
hold on;

for d=1:num_det
    smooth_plot(3,x_vals,f1scores(d,:),mark{d},'LineWidth',3);
end

print(sfigure(1),'-dpdf',sprintf('%sf1score.pdf',graph_path))

