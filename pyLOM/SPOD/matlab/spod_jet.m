%
% SPECTRAL POD
firsttry = 1;
if firsttry == 1
    clc, clear variables, close all

    %%
    % LOAD DATA
    load jetLES.mat
end
% Meshgrid
xres = 20/nx;
yres = 4/nr;
[XX,YY] = meshgrid(0:xres:20-xres,0:yres:4-yres);
%%
% % Animate pressure field
% figure(99)
% for i = 1:size(p,1)
%     contourf(XX,YY,squeeze(p(i,:,:)))
%     title('LES Jet Pressure Field')
%     xlabel('x/D')
%     ylabel('r/D')
%     pause(0.1)
% end
%%
% WINDOW = 150; % use [] to set default
% WINDOW = [128 256 512 1024 2048 4096 4900];
% nsnaps_arr = [100 150 200 250 300 400 500 600 700 800 900 1000 1500 2000 2500 3000 3500 4000 4500 5000];
nsnaps_arr = [5000];
for jj = 1:size(nsnaps_arr,2)
nsnaps = nsnaps_arr(jj); % 5000 snapshots for setting full pressure field matrix
% WINDOW = [ceil(nsnaps_arr(jj)/10)];
WINDOW = [256];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONFIGURATION FOR COMPARING TO HODMD d = 1
% WINDOW = [nsnaps-1]; % use [] to set default
% WEIGHT = []; % use [] to set default
% NOVERLAP = [WINDOW-1]; % use [] to set default
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WINDOW = [64, 128, 256, 512, 1024, 2048]; % use [] to set default
for i = 1:size(WINDOW,2)
    WEIGHT = []; % use [] to set default
    NOVERLAP = []; % use [] to set default
%     NOVERLAP = []; % use [] to set default
    p_small = p(1:nsnaps,:,:);
    tic
    [L,P,f] = spod(p_small,WINDOW(i),WEIGHT,NOVERLAP,dt);
    f = f*2.56;
    % PLOT DIFFERENT RESULTS

    %   First, we plot the SPOD spectrum.
    figure(987)
    loglog(f,L)  % L(:,1) for first energy
    xlabel('St'), ylabel('\Lambda')
%     title('First mode spectrum')
%     ylim([10e-8 10e0])
    legendInfo{i} = ['Window = ' num2str(WINDOW(i))];
    hold on

    ene_f = sum(L,2);
    tot_en = sum(sum(L));
    % Total energy
    ene_tot = sum(ene_f);
    figure(543)
    loglog(f,ene_f/tot_en)
    xlabel('St'), ylabel('SPOD total energy for each frequency')
    hold on

end
hold off
% figure(987)
% legend(legendInfo)
% figure(543)
% legend(legendInfo)
T = toc;
[M,I] = max(sum(L,2));
L2 = L;
L2(I,:) = -Inf;
[M2,I2] = max(sum(L2,2));
Omegas_arr(jj,1) = f(I);
Omegas_arr(jj,2) = f(I2);
Omegas_arr(jj,3) = T;
Omegas_arr(jj,4) = nsnaps;
end
%%
% % PLOT RESULTS
% 
% %   First, we plot the SPOD spectrum.
% figure(1)
% loglog(f,L)
% xlabel('frequency'), ylabel('SPOD mode energy')
% ylim([10e-8 10e0])
%% MODES ENERGY CONTENT
% % Energy sum for each frequency
% ene_f = sum(L,2);
% % Total energy
% ene_tot = sum(ene_f);
% energyplot = figure(2);
% figure(543)
% plot(f,ene_f)
% xlabel('frequency'), ylabel('SPOD total energy for each frequency')
% hold on
%% (1) SPOD SPECTRUM
% close all

suma = sum(sum(L)); % TOTAL ENERGY

%index = [8 13 15];
%Freq = f(index);
%Freq = f;
[M,I] = max(sum(L,2));
% index = [I(1) 1 2];
% index = [10 15 20];
% index = [3 10 13];
index = [1];
% index_ = [5 9 13 17 21]; % 256
% index_ = [5 9 13 17 21]; % 256 -- St = [0.2, 0.4, 0.6, 0.8, 1.0]
index_ = [3 13 31]; % 256 -- St = [0.1, 0.6, 1.5]
% index_ = [2 5 6 9 11]; % 128 -- St = [0.2, 0.4, 0.6, 0.8, 1.0]
% index_ = [9 17 25 33 41]; % % 512 -- St = [0.2, 0.4, 0.6, 0.8, 1.0]
% index_ = [147 164 221]; % 1500 -- St = [0.2, 0.4, 0.6, 0.8, 1.0]

Freq = f(index_);

%SPOD_ene = L(index,1)/suma;
SPOD_ene = L(index_,1)/suma;

figure11 = figure(3);
axes11 = axes('Parent',figure11);
hold(axes11,'on');
box(axes11,'on');
grid(axes11,'on');
% hold on
plot(f,L(:,1)/suma,'-b','LineWidth',3)
plot(f,L(:,2)/suma,'-r','LineWidth',3)
scatter(Freq,SPOD_ene,100,'ko','filled')
set(gca,'FontSize',18,'YScale','log')
set(gcf,'Position',[100 100 900 300]);
xlabel('f_i','FontSize',20);
ylabel('\lambda^{(k)}_{f_i}/E_t','FontSize',18);
title(['SPOD SPECTRUM'])
set(gca,'FontSize',12)
%str = {'   f_1 = 0.220','   f_2 = 0.375','   f_3 = 0.440'};
str = {"  f_1 = "+Freq(1),"  f_2 = "+Freq(2),"  f_3 = "+Freq(3)};
% text(Freq,SPOD_ene,str,'FontSize',14)

% saveas(figure11,'./Figures_paper_JFM/SPOD/Energy/C3D/plot_spodmodes','fig')
% saveas(figure11,'./Figures_paper_JFM/SPOD/Energy/C3D/plot_spodmodes','epsc')
% saveas(figure11,'./Figures_paper_JFM/SPOD/Energy/C3D/plot_spodmodes','tiff')
% saveas(figure11,'./Figures_paper_JFM/SPOD/Energy/C3D/plot_spodmodes','jpg')

%% (2) FIRST AND SECOND SPOD MODES FOR THREE FREQUENCIES
figure(4)
count = 1;
for fi = (index_)
    for mi = [1 2]
        subplot(5,2,count)
        contourf(x,r,real(squeeze(P(fi,:,:,mi))),11,'edgecolor','none'), axis equal tight, caxis(max(abs(caxis))*[-1 1])
        xlabel('x'), ylabel('r'), title(['$St=' num2str(f(fi),'%.2f$') ', mode ' num2str(mi) ', $\lambda=' num2str(L(fi,mi),'%.2g$')])
        xlim([0 14.9]); ylim([0 1.9])
        count = count + 1;
    end
end
sgtitle(['Window size: ' num2str(WINDOW(i))]) 

%%
% Plot one mode
figure;
mi = 1;
fi = index_(2);
contourf(x,r,real(squeeze(P(fi,:,:,mi))),11,'edgecolor','none'), axis equal tight, caxis(max(abs(caxis))*[-1 1])
xlabel('x'), ylabel('r'), title(['SPOD mode nº ' num2str(mi) '. St = ' num2str(f(fi))])
set(gcf, 'Position', [100 100 175*3.5 39*3.5]);
%% PLOT SPOD MODES PAPER
% close all

% COMP = 1  % STREAMWISE

fi = index(1); % FREQUENCY INDEX

% SPOD 1

%fileName_mode = strcat(folder_images, 'SPOD mode for fi = ', num2str(fi),'.png')

figure301 = figure(5);
colormap(jet)
axes301 = axes('Parent',figure301);
hold(axes301,'on');
set(gcf, 'Position', [100 100 700 350]);
%%
% a = max(max(abs(squeeze(real(P(fi,COMP,:,:,1))))));
a = max(max(abs(squeeze(real(P(fi,:,:,1))))));
% contourf(squeeze(real(P(fi,COMP,:,:,1)))'/a)
%%
contourf(XX, YY, squeeze(real(P(fi,:,:,1)))/a)
title(['SPOD MODE 1 FREQ = ',num2str(f(fi)),'; STREAMWISE COMPONENT'])
%%
set(axes301,'BoxStyle','full','CLim',[-1 1],'FontSize',18,'Layer','top');
%%
xlabel('x/D','FontSize',20);
ylabel('y/D','FontSize',20);
set(gca,'FontSize',12)
colorbar
%daspect([1 0.5 1]); colorbar
%currentImage = getframe(gcf);
%imwrite(currentImage.cdata, fileName_mode)

% saveas(figure301,'./Figures_paper_JFM/SPOD/Modes/C3D/spodmode_15_1_stream_0440','fig')
% saveas(figure301,'./Figures_paper_JFM/SPOD/Modes/C3D/spodmode_15_1_stream_0440','epsc')
% saveas(figure301,'./Figures_paper_JFM/SPOD/Modes/C3D/spodmode_15_1_stream_0440','tiff')
% saveas(figure301,'./Figures_paper_JFM/SPOD/Modes/C3D/spodmode_15_1_stream_0440','jpg')

%%
% SPOD 2
figure302 = figure(6);
colormap(jet)
axes302 = axes('Parent',figure302);
hold(axes302,'on');
set(gcf, 'Position', [100 100 700 350]);
% a = max(max(abs(squeeze(real(P(fi,COMP,:,:,2))))));
a = max(max(abs(squeeze(real(P(fi,:,:,2))))));
% contourf(squeeze(real(P(fi,COMP,:,:,2)))'/a)
contourf(XX, YY, squeeze(real(P(fi,:,:,2)))/a)
title(['SPOD MODE 2 FREQ = ',num2str(f(fi)),'; STREAMWISE COMPONENT'])
set(axes302,'BoxStyle','full','CLim',[-1 1],'FontSize',18,'Layer','top');
xlabel('x/D','FontSize',20);
ylabel('y/D','FontSize',20);
set(gca,'FontSize',12)
colorbar
% saveas(figure302,'./Figures_paper_JFM/SPOD/Modes/C3D/spodmode_15_2_stream_0440','fig')
% saveas(figure302,'./Figures_paper_JFM/SPOD/Modes/C3D/spodmode_15_2_stream_0440','epsc')
% saveas(figure302,'./Figures_paper_JFM/SPOD/Modes/C3D/spodmode_15_2_stream_0440','tiff')
% saveas(figure302,'./Figures_paper_JFM/SPOD/Modes/C3D/spodmode_15_2_stream_0440','jpg')


%%
% ANIMATE MODES

% figure(301)
% nt      = 30;
% T       = 1/f(10);              % period of the 10th frequency
% time    = linspace(0,T,nt);     % animate over one period
% count = 1;
% for ti = 1:nt
%     for fi = [8 13 15]
%         for mi = [1 2]
%             subplot(3,2,count)
%             pcolor(nx,nr,real(squeeze(P(fi,:,:,mi)*exp(2i*pi*f(fi)*time(ti))))'), shading interp, axis equal tight, caxis(max(abs(caxis))*[-1 1])
%             xlabel('x'), ylabel('r'), title(['f=' num2str(f(fi),'%.2f') ', mode ' num2str(mi) ', \lambda=' num2str(L(fi,mi),'%.2g')])
%             count = count + 1;
%             hold on
%         end
%     end
%     drawnow
%     hold off
%     count = 1;
% end
n_f = size(f,2)-1

%%
a = max(max(abs(squeeze(real(P(fi,:,:,1))))));
figure15 = figure(7)
subplot(2,1,1)
contourf(XX, YY, squeeze(real(P(fi,:,:,1)))/a)
title(['SPOD MODE 1 FREQ = ',num2str(f(fi)),'; STREAMWISE COMPONENT'])
%%
figure(7)
subplot(2,1,2)
axes11 = axes('Parent',figure11);
hold(axes11,'on');
%%
box(axes11,'on');
grid(axes11,'on');
hold on
%%
plot(f,L(:,1)/suma,'-b','LineWidth',3)
plot(f,L(:,2)/suma,'-r','LineWidth',3)
scatter(Freq,SPOD_ene,100,'ko','filled')
set(gca,'FontSize',18,'YScale','log')
set(gcf,'Position',[100 100 900 300]);
xlabel('f_i','FontSize',20);
ylabel('\lambda^{(k)}_{f_i}/E_t','FontSize',18);
title(['SPOD SPECTRUM'])
set(gca,'FontSize',12)
%str = {'   f_1 = 0.220','   f_2 = 0.375','   f_3 = 0.440'};
str = {"  f_1 = "+Freq(1),"  f_2 = "+Freq(2),"  f_3 = "+Freq(3)};
text(Freq,SPOD_ene,str,'FontSize',14)

%%
en_fit = ene_f(13:end)';
f_fit = f(13:end);
