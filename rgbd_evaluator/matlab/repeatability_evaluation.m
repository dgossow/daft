function seqrepeat = repeatability_evaluation( base_path, dataset_name, det_suffix, x_val_file )

data_path = [base_path, dataset_name, '/'];
graph_path = [data_path, 'results/'];
mkdir(graph_path)

mark={'-rs';'--bp';'-kx';'--kv';':r+';'-.bp';'--b>'};
num_det = size(det_suffix,1);
x_vals = load( sprintf( '%s%s', data_path, x_val_file ) );
num_img = size( x_vals, 2 ) + 1;

figure(1);clf;
axes('LineWidth',3);
set(gca,'FontSize',35)
grid on;
ylabel('repeatability %')
xlabel(x_val_file);
hold on;
figure(2);clf;
axes('LineWidth',3);
set(gca,'FontSize',35)
grid on;
ylabel('nb of correspondences')
xlabel(x_val_file);
hold on;

for d=1:num_det
    
    seqrepeat=[];
    seqcorresp=[];
    
    for i=2:num_img
        
        file1=sprintf('%simg1.%s',data_path,char(det_suffix(d)));
        file2=sprintf('%simg%d.%s',data_path,i,char(det_suffix(d)));
        Hom=sprintf('%sH1to%dp',data_path,i);
        imf1=sprintf('%simg1.ppm',data_path);
        imf2=sprintf('%simg%d.ppm',data_path,i);
        
        [erro,repeat,corresp, match_score,matches, twi]=repeatability(file1,file2,Hom,imf1,imf2, 1);
        seqrepeat=[seqrepeat repeat(4)];
        seqcorresp=[seqcorresp corresp(4)];
    end
    
    mark{d}
    figure(1);  plot(x_vals,seqrepeat,mark{d},'LineWidth',3);
    figure(2);  plot(x_vals,seqcorresp,mark{d},'LineWidth',3);

end

figure(1)
axis([x_vals(1) x_vals(num_img-1) 30 100]);
%axis 'auto x'

figure(2)
axis([x_vals(1) x_vals(num_img-1) 0 750]);

for f=1:2
    figure(f);
end

print(figure(1),'-dpdf',sprintf('%srepeatability.pdf',graph_path))
print(figure(2),'-dpdf',sprintf('%snum_correspondences.pdf',graph_path))

figure(1)
legend(det_suffix,'Location','NorthOutside');
print(figure(1),'-dpdf',sprintf('%slegend.pdf',graph_path))

