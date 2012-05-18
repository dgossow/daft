function [ x_smooth, y_smooth ] = smooth_data( X, Y, gauss_sigma )
%smooth evaluation data using a gaussian

 %Y=[87.6238   83.6634   83.1683   84.1584   78.7129   60.3960   82.6733   59.0361   56.4286   55.9524   44.7368   45.2381   54.0541   50.0000];
 %X=[42.3534	45.611	43.5781	46.7971	49.4403	53.9644	52.1427	60.6864	61.8536	60.5621	67.1328	68.3104	68.5689	75.8563	];

 A=sortrows([X',Y']);
 
 min_x = min(X);
 max_x = max(X);
 
 num_values = size(X,2);
 gauss_sigma = (max_x - min_x) / 10;%num_values;% * 1.5;
 
 num_bins = 100;
 buffer_width = (max_x-min_x) * (num_bins+1)/num_bins;
 buffer_res = (num_bins) / buffer_width;
 
 num_entries = zeros( num_bins, 1 );
 x_smooth = zeros( num_bins, 1 );
 y_smooth = zeros( num_bins, 1 );
 
 for i=1:size(Y,2)
     x = A(i,1);
     y = A(i,2);
     x_bin = round((x - min_x) * buffer_res) + 1;
     if ~isnan(x_bin) && ~isnan(y)
         %if x_bin == 0
             %x_bin = 1
         %end
         num_entries(x_bin,1) = num_entries(x_bin,1)+1;
         %x_vals(x_bin,1) = x_vals(x_bin,1)+x;
         y_smooth(x_bin,1) = y_smooth(x_bin,1)+y;
     end
 end
 
 for i=1:num_bins
     x_smooth(i,1) = (i-1)/(num_bins-1)*(max_x-min_x) + min_x;
 end
 
 %num_entries
 %y_smooth
 
 % make gauss
 %gauss_sigma = 4;
 gauss_size = num_bins*2;
 x_bin = linspace(-gauss_size / 2, gauss_size / 2, gauss_size);
 gaussFilter = exp(-x_bin .^ 2 / (2 * (gauss_sigma * buffer_res) ^ 2));
 gaussFilter = gaussFilter / sum (gaussFilter); % normalize
 
 num_entries = conv (num_entries, gaussFilter, 'same');
 y_smooth = conv (y_smooth, gaussFilter, 'same') ./ num_entries;

 if false
     figure(1);
     clf;
     hold on;
     plot( A(:,1), A(:,2) );
     plot( x_smooth, y_smooth, 'r' );
 end
 
 %num_entries = filter (gaussFilter,1, num_entries)
 %y_vals = filter (gaussFilter,1, y_vals)
  

end

