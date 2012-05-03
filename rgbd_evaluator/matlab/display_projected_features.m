function display_projected_features(imf1,feat1,feat2)
%
%Displays affine regions on the image
%
%disp_features(file1,imf1,dx,dy)
%
%file1 - 'filename' - ASCII file with affine regions
%imf1 - 'filename'- image
%dx - shifts all the regions in the image by dx  
%dy - shifts all the regions in the image by dy
%
%example:
%disp_features('img1.haraff','img1.ppm',0,0)

clf;imshow(imf1);

for c1=1:size(feat2,2)
    drawellipse([feat2(3,c1) feat2(4,c1); feat2(4,c1) feat2(5,c1) ], feat2(1,c1), feat2(2,c1),'y',0);
end
for c1=1:size(feat1,2)
    drawellipse([feat1(3,c1) feat1(4,c1); feat1(4,c1) feat1(5,c1) ], feat1(1,c1), feat1(2,c1),'g',0);
end

end

function drawellipse(Mi,i,j,col,make_circle)
hold on;
[v e]=eig(Mi);

l1=1/sqrt(e(1));
l2=1/sqrt(e(4));

if make_circle
    radius=sqrt(l1*l2);
    l1=radius;
    l2=radius;
end

alpha=atan2(v(4),v(3));
s=1;
t = 0:pi/50:2*pi;
y=s*(l2*sin(t));
x=s*(l1*cos(t));

xbar=x*cos(alpha) + y*sin(alpha);
ybar=y*cos(alpha) - x*sin(alpha);
plot(ybar+i,xbar+j,'-k','LineWidth',3);
plot(ybar+i,xbar+j,col,'LineWidth',1);
%col='-k';
%plot([i-2 i+2],[j j],col,'LineWidth',3);
%plot([i i],[j-2 j+2],col,'LineWidth',3);
set(gca,'Position',[0 0 1 1]);
hold off;

end
