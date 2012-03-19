function [precision, recall] = descperf_evaluation( base_path, dataset_name, det_suffix, x_val_file, num_img )

data_path = [base_path, dataset_name, '/'];
graph_path = [data_path, 'results/'];

mkdir(graph_path)

fprintf(1,'It may take a while, i.e. 30min \n');

num_det = size(det_suffix,1);

mark={'-rs';'--bp';'-kx';'--kv';':r+';'-.bp';'--b>'};
x_vals = load( sprintf( '%s%s', data_path, x_val_file ) );

for i=2:num_img

    figure(1);clf;
    set(gca,'FontSize',17)
    grid on;
    ylabel('recall')
    xlabel('1 - precision');
    hold on;
    
    for d=1:num_det
    
        file1=sprintf('%simg1.%s',data_path,char(det_suffix(d)));
        file2=sprintf('%simg%d.%s',data_path,i,char(det_suffix(d)));
        Hom=sprintf('%sH1to%dp',data_path,i);
        imf1=sprintf('%simg1.ppm',data_path);
        imf2=sprintf('%simg%d.ppm',data_path,i);

        [erro,repeat,corresp, match_score,matches, twi]=repeatability(file1,file2,Hom,imf1,imf2, 0);

        [correct_match_nn,total_match_nn,correct_match_sim,total_match_sim,correct_match_rn,total_match_rn] = descperf(file1,file2,Hom,imf1,imf2,corresp(4),twi);

        recall=correct_match_rn./corresp(4)
        precision=(total_match_rn-correct_match_rn)./total_match_rn

        plot( precision, recall, mark{d} );
    end

    axis([0 0.5 0 1]);

    figname = sprintf('%s%s_descperf %s %f.pdf',graph_path, dataset_name, x_val_file, x_vals(i-1) );
    legend(det_suffix,'Location','NorthOutside');
    print(figure(1),'-dpdf',figname)

end

