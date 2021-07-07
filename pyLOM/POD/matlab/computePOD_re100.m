clear all, close all, clc
load CYLINDER_ALL.mat

% DATA 
% VORTALL: vorticity field
% UALL: velocity Ux
% VALL: velocity Uy
N = 151;
X = UALL(:,1:N);

%% compute POD after subtracting mean (i.e., do PCA)
Uavg = mean(X,2);
X_m = X-Uavg*ones(1,size(X,2));
Y = X_m;
tic
[PSI,S,V] = svd(Y,'econ');
% PSI are POD modes
% figure
% semilogy(diag(S)./sum(diag(S)),'o'); % plot singular vals
T = toc;
% % for k=1:12 % plot first 12 POD modes
% %     f1 = plotCylinder(reshape(PSI(:,k),nx,ny),nx,ny);
% % end
%%
figure(500)
n_snaps = 1:N;
accumulative_S = zeros(1,N);
diag_S = diag(S);

for i = n_snaps
    accumulative_S(i) = norm(diag_S(i:N),2)/norm((diag_S),2);
end

semilogy(n_snaps,accumulative_S, 'bo')
ylabel('varepsilon1')
xlabel('Truncation size')
title('Tolerance')
ylim([0 1])
hold on
%% V representation
dt = 0.2;
t = (dt:1:size(V,2))*dt;
m = 1; % POD temporal mode number
y = V(:,m);
figure(1)
subplot(3,1,2)
plot(t,y(:,1),'b','LineWidth',1.2)
title(['POD Temporal mode nº ' num2str(m)])

%% Fast Fourier Transform of V
Y = fft(y,N); % Fast discrete Fourier Transform
PSD = Y.*conj(Y)/N; % power spectrum (how much power is in each freq)
freq = 1/(dt*N)*(0:N); % creates the x-axis of freqs in Hz
L = 1:floor(N/2);
figure(1)
subplot(3,1,3)
plot(freq(L),PSD(L))
title('Power Spectrum')
xlabel('St')

%%
deltaX=(8+1)/448;
deltaY=(2+2)/198;
X=[0:1:448]*deltaX-1;
Y=[0:1:198]*deltaY-2;
[XX, YY]=meshgrid(X,Y);
PSI_=reshape(PSI(:,m),[nx ny]);
figure(1)
subplot(3,1,1)
contourf(XX,YY,squeeze(PSI_(:,:)))
title(['POD mode nº ' num2str(m)])
% colorbar('southoutside')

%%
% for k=1:1 % plot first 12 POD modes
%     f1 = plotCylinder(reshape(PSI(:,k),nx,ny),nx,ny);
% end