function f1 = plotCylinder(VORT,ny,nx)

f1 = figure
vortmin = -5;  % only plot what is in -5 to 5 range
vortmax = 5;
VORT(VORT>vortmax) = vortmax;  % cutoff at vortmax
VORT(VORT<vortmin) = vortmin;  % cutoff at vortmin

imagesc(VORT); % plot vorticity field
load CCcool.mat 
colormap(CC);  % use custom colormap

% clean up axes
set(gca,'XTick',[1 50 100 150 200 250 300 350 400 449],'XTickLabel',{'-1','0','1','2','3','4','5','6','7','8'})
set(gca,'YTick',[1 50 100 150 199],'YTickLabel',{'2','1','0','-1','-2'});
set(gcf,'Position',[100 100 300 130])
axis equal
hold on

% add contour lines (positive = solid, negative = dotted)
contour(VORT,[-5.5:.5:-.5 -.25 -.125],':k','LineWidth',1.2)
contour(VORT,[.125 .25 .5:.5:5.5],'-k','LineWidth',1.2)

theta = (1:100)/100'*2*pi;
x = 49+25*sin(theta);
y = 99+25*cos(theta);
fill(x,y,[.3 .3 .3])  % place cylinder
plot(x,y,'k','LineWidth',1.2) % cylinder boundary

set(gcf,'PaperPositionMode','auto') % 
% print('-depsc2', '-loose', 'figures/cylinder'); % eps are vector images
% fix_lines('figures/cylinder.eps','figures/cylinder.eps') % fix dashed lines