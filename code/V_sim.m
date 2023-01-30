%
% Compute value functions through consumption simulation
%


clear; close all;
load("results\results_s14z72_densea.mat");


rng(12346);
dB = B_v(2) - B_v(1);
dN = N_v(2) - N_v(1);
da = a(2,1,1,1) - a(1,1,1,1);
n_az = n_a * n_z;

%% Value Function Simulation
n_simC = 10;
t_simC = 2000;
p_simC = 100;
dt_simC = .1;

% Create a grid
a_sim = a(:,:,1,1);
z_sim = z(:,:,1,1);

% Consumption Simulation
a_now = 0.16 .* ones(p_simC,n_simC);
z_now = z1 .* ones(p_simC,n_simC);
B_now = linspace(B_min+0.1,B_max-0.1,n_simC) .* ones(p_simC,1);
N_now = 3.2 .* ones(p_simC,n_simC);
V_true = mean(interpn(a,z,B,N,V,a_now,z_now,B_now,N_now));

prob_z1z1 = exp(- lambda1 * dt_simC);
prob_z2z1 = 1 -  exp(- lambda2 * dt_simC);

V_psims = zeros(p_simC,n_simC);
t_cum = 0;

tic
for i_tsim = 1:t_simC
    C_now = interpn(a,z,B,N,c,a_now,z_now,B_now,N_now);
    s_now = interpn(a,z,B,N,s,a_now,z_now,B_now,N_now);

    % next period B
    B_next = B_now + interpn(a,z,B,N,h,a_now,z_now,B_now,N_now) .* dt_simC;

    % next period N
    muN_now = interpn(a,z,B,N,muN,a_now,z_now,B_now,N_now);
    sigN_now = interpn(a,z,B,N,sigN,a_now,z_now,B_now,N_now);
    N_next = N_now + muN_now .* dt_simC + sigN_now .* dt_simC ^ 0.5 .* randn(p_simC,n_simC);

    % next period a
    a_next = a_now + s_now .* dt_simC;

    % next period z
    prob_z1_next = prob_z1z1 .* ones(p_simC,n_simC);
    prob_z1_next(z_now == z2) = prob_z2z1;
    z1_next = unifrnd(0,1,p_simC,n_simC) <= prob_z1_next;
    z_next = z1_next * z1 + (1 - z1_next) * z2;
    
    % update
    a_now = a_next;
    z_now = z_next;
    B_now = B_next;
    B_now(B_now > B_max) = B_max;
    B_now(B_now < B_min) = B_min;
    N_now = N_next;
    N_now(N_now > N_max) = N_max;
    N_now(N_now < N_min) = N_min;

    V_psims = V_psims + exp(-rho * t_cum) * ...
        (C_now .^ (1 - gam) - 1) ./ (1 - gam) .* dt_simC;
    t_cum = t_cum + dt_simC;
end

toc
V_sims = mean(V_psims);
plot(linspace(B_min+0.1,B_max-0.1,n_simC),V_sims,'r--');
hold on;
plot(linspace(B_min+0.1,B_max-0.1,n_simC),V_true,'k-');
xlabel("$B$", "interpreter", "latex")
ylabel("$V$", "interpreter", "latex")
legend("Simulation", "Numerical");

set(gcf,'position',[300,300,700,400])
saveas(gcf, "..\report\graphs\V_error_up", "epsc");













