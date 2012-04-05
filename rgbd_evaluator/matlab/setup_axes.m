function [ output_args ] = setup_axes( x_vals, num_img, y_range )

fprintf('x axis: %f %f\n', x_vals(1), x_vals(num_img-1));

x1 = min(x_vals(1), x_vals(num_img-1));
x2 = max(x_vals(1), x_vals(num_img-1));

xdir = 'normal';
if x_vals(1) > x_vals(num_img-1)
    xdir ='reverse';
end

axis([x1 x2 y_range]);

set(gca,'XDir',xdir);

end

