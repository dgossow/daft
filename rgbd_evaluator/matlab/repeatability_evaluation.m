function repeatability_evaluation( base_path, dataset_name, det_suffix, x_val_file, num_img )

data_path = [base_path, dataset_name, '/'];

fprintf(1,'It may take a while, i.e. 30min \n');
%mex c_eoverlap.cxx;

%det_suffix=['haraff';'hesaff';'mseraf';'ibraff';'ebraff'];


figure(1);clf;
grid on;
ylabel('repeatability %')
xlabel(x_val_file);
hold on;
figure(2);clf;
grid on;
ylabel('nb of correspondences')
xlabel('viewpoint angle');
hold on;

mark=['-kx';'-rv';'-gs';'-m+';'-bp'];

num_det = size(det_suffix,1);

x_vals = load( sprintf( '%s%s', data_path, x_val_file ) );

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
    
    figure(1);  plot(x_vals,seqrepeat,mark(d,:));
    figure(2);  plot(x_vals,seqcorresp,mark(d,:));

end

figure(1)
axis([10 70 0 100]);
axis 'auto x'

for f=1:2
    figure(f);
    legend(det_suffix);
end

print(figure(1),'-dpdf',sprintf('%s%s_repeatability.pdf',base_path, dataset_name))
print(figure(2),'-dpdf',sprintf('%s%s_num_correspondences.pdf',base_path, dataset_name))
