function setup_axes( x_vals, y_range )

num_xvals = size(x_vals,2);

%fprintf('x axis: %f %f\n', x_vals(1), x_vals(num_img-1));

x1 = min(x_vals(1), x_vals(num_xvals));
x2 = max(x_vals(1), x_vals(num_xvals));

xdir = 'normal';
if x_vals(1) > x_vals(num_xvals)
    xdir ='reverse';
end

axis([x1 x2 y_range]);

set(gca,'XDir',xdir);

end

