function [precision, recall] = descperf_evaluation( base_path, dataset_name, det_suffix, x_val_file )

data_path = [base_path, dataset_name, '/'];
graph_path = [data_path, 'results/'];
mkdir(graph_path)

fprintf(1,'It may take a while, i.e. 30min \n');

num_det = size(det_suffix,1);

mark={'-rp';'--ks';'-.kx';'--kv';':r+';'-.bp';'--b>'};

%mark={'-rs';'--bp';'-kx';'--kv';':r+';'-.bp';'--b>'};

x_vals = load( sprintf( '%s%s', data_path, x_val_file ) );
num_img = size( x_vals, 2 ) + 1;

for i=2:num_img

    sfigure(1);clf;
    axes('LineWidth',3);
    set(gca,'FontSize',35);
    %set(gca,'XTick',[0 0.1 0.2 0.3 0.4 0.5]);
    set(gca,'XTick',[0.0 0.25 0.5 0.75 1.0]);
    set(gca,'YTick',[0.0 20.0 40.0 60.0 80.0 100.0]);
    grid on;
    ylabel('recall %');
    xlabel('precision %');
    title( sprintf('%s = %1.1f', x_val_file, x_vals(i-1) ) );

    
    hold on;
    
    for d=1:num_det
    
        file1=sprintf('%simg1.%s',data_path,char(det_suffix(d)));
        file2=sprintf('%simg%d.%s',data_path,i,char(det_suffix(d)));
        Hom=sprintf('%sH1to%dp',data_path,i);
        imf1=sprintf('%simg1.ppm',data_path);
        imf2=sprintf('%simg%d.ppm',data_path,i);

        [erro,repeat,corresp, match_score,matches, twi]=repeatability(file1,file2,Hom,imf1,imf2,'',0);

        [correct_match_nn,total_match_nn,correct_match_sim,total_match_sim,correct_match_rn,total_match_rn] = descperf(file1,file2,Hom,imf1,imf2,corresp(4),twi);

        recall=correct_match_rn./corresp(4)*100;
        %precision=(total_match_rn-correct_match_rn)./total_match_rn
        precision=correct_match_rn./total_match_rn*100;

        plot( precision, recall, mark{d},'LineWidth',3 );
    end

    axis([0 100 0 100]);
    figname = sprintf('%sdescperf_%i.pdf',graph_path, i-1 );
    print(sfigure(1),'-dpdf',figname);
end

