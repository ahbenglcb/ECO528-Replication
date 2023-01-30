%
% To replicate the phase diagram in the paper
%

clear; 
clf; close all;

load("results\results_s14z72_denseab.mat");
rnn = alpha .* (Bnn + Nnn) .^ (alpha - 1) - delta - sig .^ 2 .* (Bnn + Nnn) ./ Nnn;
muNnn = alpha .* (Bnn + Nnn) .^ alpha - delta .* (Bnn + Nnn) ...
    - rnn .* Bnn - rhohat .* Nnn;

hold on;
Mh = contour(Bnn,Nnn,h_fine,[0 0],"k-");
MN = contour(Bnn,Nnn,muNnn,[0 0],"r--");
surf(Bnn,Nnn,Y_ave,'edgecolor','green','facecolor',"none")
xlabel("$B$", "interpreter", "latex")
ylabel("$N$", "interpreter", "latex")
legend("$h(B,N) = 0$", "$\mu^N(B,N) = 0$", "Ergodic dist.", "interpreter", "latex")

set(gcf,'position',[300,300,700,400])
saveas(gcf, "..\report\graphs\sss_upwind_denseab", "epsc");

