%
% To generate some plots for report
%

%% Value function without upwind scheme
clear; close all;
load("results\results_s14z72_noupwind");

surf(reshape(B(1,1,:,:),n_B,n_N),reshape(N(1,1,:,:),n_B,n_N),reshape(V(5,1,:,:),n_B,n_N));
xlabel("$B$", "interpreter", "latex")
ylabel("$N$", "interpreter", "latex")
zlabel("$V$", "interpreter", "latex")

set(gcf,'position',[300,300,700,400])
saveas(gcf, "..\report\graphs\V_noupwind", "epsc");

%% Value function without upwind scheme and denser grid
clear; close all;
load("results\results_broken");

surf(reshape(B(1,1,:,:),n_B,n_N),reshape(N(1,1,:,:),n_B,n_N),reshape(V(5,1,:,:),n_B,n_N));
xlabel("$B$", "interpreter", "latex")
ylabel("$N$", "interpreter", "latex")
zlabel("$V$", "interpreter", "latex")

set(gcf,'position',[300,300,700,400])
saveas(gcf, "..\report\graphs\V_broken", "epsc");

%% Value function with upwind scheme
clear; close all;
load("results\results_s14z72_densea");

surf(reshape(B(1,1,:,:),n_B,n_N),reshape(N(1,1,:,:),n_B,n_N),reshape(V(5,1,:,:),n_B,n_N));
xlabel("$B$", "interpreter", "latex")
ylabel("$N$", "interpreter", "latex")
zlabel("$V$", "interpreter", "latex")

set(gcf,'position',[300,300,700,400])
saveas(gcf, "..\report\graphs\V_up", "epsc");



%% Phase diagrams sigmas
clear; 
clf; close all;

sig_plot = {"05","08","10","12","14"};
for i_plot = 1:5
    load(join(["results\results_s",sig_plot{i_plot},"z72.mat"],""));
    rnn = alpha .* (Bnn + Nnn) .^ (alpha - 1) - delta - sig .^ 2 .* (Bnn + Nnn) ./ Nnn;
    muNnn = alpha .* (Bnn + Nnn) .^ alpha - delta .* (Bnn + Nnn) ...
        - rnn .* Bnn - rhohat .* Nnn;

    subplot(2,3,i_plot)
    hold on;
    Mh = contour(Bnn,Nnn,h_fine,[0 0],"k-");
    MN = contour(Bnn,Nnn,muNnn,[0 0],"r--");
    surf(Bnn,Nnn,Y_ave,'edgecolor','green','facecolor',"none")
    xlabel("$B$", "interpreter", "latex")
    ylabel("$N$", "interpreter", "latex")
    title(join(["$\sigma = ",sig,"$"],""), "interpreter", "latex")
    
end

h = legend("$h(B,N) = 0$", "$\mu^N(B,N) = 0$", "Ergodic dist.", "interpreter", "latex");
rect = [0.77, 0.13, .1, .1];
set(h, 'Position', rect);
set(gcf,'position',[300,300,700,400])
saveas(gcf, "..\report\graphs\sss_sig", "epsc");

%% Phase diagrams zs
clear; 
clf; close all;

z_plot = {"60","65","72","75","80","85"};
for i_plot = 1:6
    load(join(["results\results_s12z",z_plot{i_plot},".mat"],""));
    rnn = alpha .* (Bnn + Nnn) .^ (alpha - 1) - delta - sig .^ 2 .* (Bnn + Nnn) ./ Nnn;
    muNnn = alpha .* (Bnn + Nnn) .^ alpha - delta .* (Bnn + Nnn) ...
        - rnn .* Bnn - rhohat .* Nnn;

    subplot(2,3,i_plot)
    hold on;
    Mh = contour(Bnn,Nnn,h_fine,[0 0],"k-");
    MN = contour(Bnn,Nnn,muNnn,[0 0],"r--");
    surf(Bnn,Nnn,Y_ave,'edgecolor','green','facecolor',"none")
    xlabel("$B$", "interpreter", "latex")
    ylabel("$N$", "interpreter", "latex")
    title(join(["$z = ",z1,"$"],""), "interpreter", "latex")
    
end

h = legend("$h(B,N) = 0$", "$\mu^N(B,N) = 0$", "Ergodic dist.", "interpreter", "latex");
rect = [0.77, 0.33, .1, .1];
set(h, 'Position', rect);
set(gcf,'position',[300,300,700,400])
saveas(gcf, "..\report\graphs\sss_z", "epsc");





