%
% ECO 528 Replication Project
% Financial Frictions and the Wealth Distribution
% by Fernandez-Villaverde, Hurtado and Nuno
%

clear; close all;
addpath("functions")
% delete(gcp('nocreate'));
% parpool(8);
seed_base = 12345;

% Parameters
alpha = 0.35;                               % capital share
delta = 0.1;                                % capital depreciation
gam = 2;                                    % risk aversion
rho = 0.05;                                 % household discount rate
lambda1 = 0.986;                            % tran rate from unemp-to-emp
lambda2 = 0.052;                            % tran rate from emp-to-unemp  
z1 = 0.72;                                  % income in unemp state
z2 = 1 + lambda2 / lambda1 * (1 - z1);      % income in emp state
rhohat = 0.04971;                           % expert discount rate
sig = 0.014;                                % capital vol

% Grids
n_z = 2;
n_a = 501; a_min = 0; a_max = 15;
n_B = 7; B_min = 0.7; B_max = 2.5;
n_N = 51; N_min = 1.2; N_max = 3.2;

% Neural network
n_Q = 16; lr_theta = 1;
n_Bnn = 101; n_Nnn = 101;
n_tnn = 1000; lr_h = 0.1;

% Value function iteration 
dt = 1000; lr = 0.5; tol_V = 1e-5;

% KFE and Simulation
dt_sim = 1 / 12; n_sims = 2000 / dt_sim;
n_burnin = 1000;

% Generate par
par.alpha = alpha; 
par.delta = delta;
par.gam = gam;
par.rho = rho;
par.lambda1 = lambda1;
par.lambda2 = lambda2;
par.z1 = z1;
par.z2 = z2;
par.rhohat = rhohat;
par.sig = sig;

par.n_z = n_z;
par.n_a = n_a; par.a_min = a_min; par.a_max = a_max;
par.n_B = n_B; par.B_min = B_min; par.B_max = B_max;
par.n_N = n_N; par.N_min = N_min; par.N_max = N_max;

par.n_Q = n_Q; par.lr_theta = lr_theta;
par.n_Bnn = n_Bnn; par.n_Nnn = n_Nnn;
par.n_tnn = n_tnn; par.lr_h = lr_h;

par.n_burnin = n_burnin;

par.dt = dt; par.lr = lr; par.tol_V = tol_V;

par.dt_sim = dt_sim; par.n_sims = n_sims;

% Grids [Dim (1,2,3,4) = (a,z,B,N)]
a_v = linspace(par.a_min, par.a_max, par.n_a)';
z_v = [par.z1; par.z2];
B_v = linspace(par.B_min, par.B_max, par.n_B)';
N_v = linspace(par.N_min, par.N_max, par.n_N)';
[a, z, B, N] = ndgrid(a_v, z_v, B_v, N_v);

da = a_v(2) - a_v(1);
dB = B_v(2) - B_v(1);
dN = N_v(2) - N_v(1);

% Neural grids
Bnn_v = linspace(par.B_min, par.B_max, par.n_Bnn)';
Nnn_v = linspace(par.N_min, par.N_max, par.n_Nnn)';
[Bnn, Nnn] = meshgrid(Bnn_v, Nnn_v);
Bnn = Bnn';
Nnn = Nnn';

dBnn = Bnn_v(2) - Bnn_v(1);
dNnn = Nnn_v(2) - Nnn_v(1);

lambda = zeros(n_a, n_z, n_B, n_N);
lambda(z == z1) = lambda1;
lambda(z == z2) = lambda2;

n_azBN = n_a * n_z * n_B * n_N;

%% Steady state
steady_state;

%% Value function iteration
% Initial guess: Perceived Law of Motion (see p. 18)
n_theta = 1 + 4 * n_Q;
theta = zeros(n_theta,1);
theta_old = theta;

% compute wage and r
w = (1 - alpha) .* (B + N) .^ alpha;
r = alpha .* (B + N) .^ (alpha - 1) - delta - sig .^ 2 .* (B + N) ./ N;

% Compute muN and sigN
muN = alpha .* (B + N) .^ alpha - delta .* (B + N) ...
    - r .* B - rhohat .* N;
sigN = sig .* (B + N);
    
if min(muN(1,1,:,1)) < 0
    fprintf("muN < 0 at N_min \n");
    keyboard;
end

Initial guess: V
V_old = (w .* z + r .* a) .^ (1 - gam) / (1 - gam) / rho;
V = V_old;

% Compute h using NN
h = reshape(plm([B(:),N(:)],theta,@softplus),n_a,n_z,n_B,n_N);
h_old = h;
h_fine_old = reshape(plm([Bnn(:),Nnn(:)],theta,@softplus),n_Bnn,n_Nnn);

time_start = tic;
for i_iter = 1:1000


    time_start_V = tic;
    converge = "No";
    for i_V = 1:10000
        % Derivatives
        dV_af = zeros(n_a, n_z, n_B, n_N); dV_ab = zeros(n_a, n_z, n_B, n_N);
        dV_B = zeros(n_a, n_z, n_B, n_N); dV_N = zeros(n_a, n_z, n_B, n_N);
        dV_NN = zeros(n_a, n_z, n_B, n_N);

        dV_af(1:end-1,:,:,:) = ( V(2:end,:,:,:) - V(1:end-1,:,:,:) ) ./ da;
        dV_af(end,:,:,:) = (w(end,:,:,:) .* z(end,:,:,:) ...
            + r(end,:,:,:) .* a(end,:,:,:) ) .^ (- gam) ;
        dV_ab(2:end,:,:,:) = ( V(2:end,:,:,:) - V(1:end-1,:,:,:) ) ./ da;
        dV_ab(1,:,:,:) = (w(1,:,:,:) .* z(1,:,:,:) ...
            + r(1,:,:,:) .* a(1,:,:,:) ) .^ (- gam) ;

        dV_B(:,:,2:end,:) = ( V(:,:,2:end,:) - V(:,:,1:end-1,:) ) ./ dB;
        dV_B(:,:,1,:) = dV_B(:,:,2,:);

        dV_N(:,:,:,2:end) = ( V(:,:,:,2:end) - V(:,:,:,1:end-1) ) ./ dN;
        dV_N(:,:,:,1) = dV_N(:,:,:,2);

        dV_NN(:,:,:,2:end-1) = ( V(:,:,:,3:end) - 2 .* V(:,:,:,2:end-1) ...
            + V(:,:,:,1:end-2) ) ./ dN.^2;
        dV_NN(:,:,:,[1 end]) = dV_NN(:,:,:,[2 end-1]);

        % Consumption, forward
        c_f = dV_af .^ (- 1 / gam);
        s_f = w .* z + r .* a - c_f;

        % Consumption, backward
        c_b = dV_ab .^ (- 1 / gam);
        s_b = w .* z + r .* a - c_b;

        % Consumption, undetermined sign
        c_u = w .* z + r .* a;
        s_u = zeros(n_a,n_z,n_B,n_N);
        dV_au = c_u .^ (- gam);

        ind_f = s_f > 0;
        ind_b = s_b < 0;
        ind_u = 1 - ind_f - ind_b;

        dV_a = ind_f .* dV_af + ind_b .* dV_ab + ind_u .* dV_au;
        c = dV_a .^ (-1 / gam);
        s = w .* z + r .* a - c;        

        % Compute matrices
        beta = - abs(s) ./ da - lambda - abs(h) ./ dB - abs(muN) ./ dN - (sigN ./ dN) .^ 2;
        varkappa = 0.5 .* (sigN ./ dN) .^ 2;
        muN_f = max(muN,0) ./ dN;
        muN_b = max(-muN,0) ./ dN;
        varrho = 0.5 .* (sigN ./ dN) .^ 2;

        % Generate matrices A, B and vector d
        A = sparse(n_azBN,n_azBN);

        n_az = n_a * n_z;
        n_azB = n_az * n_B;
        for i_N = 1:n_N
            A_N = sparse(n_azB,n_azB);

            for i_B = 1:n_B

                % Fill A_BN
                beta_now = beta(:,:,i_B,i_N);
                s_now = s(:,:,i_B,i_N);
                A_BN = spdiags(beta_now(:),0,n_az,n_az) ...
                    + spdiags(max(s_now(:) ./ da, 0),-1,n_az,n_az)' ...
                    + spdiags(max(- s_now(:) ./ da, 0),1,n_az,n_az)' ...
                    + spdiags(lambda1*ones(n_a,1),-n_a,n_az,n_az)' ...
                    + spdiags(lambda2*ones(n_a,1),-n_a,n_az,n_az);

                % Fill A_N
                ind_azB = (i_B - 1) * n_az + 1:i_B * n_az;
                h_now = h(1,1,i_B,i_N);
                if i_B == 1
                    A_N(ind_azB, ind_azB) = A_BN;
                    A_N(ind_azB, ind_azB+n_az) = abs(h_now) ./ dB .* speye(n_az);          
                elseif i_B == n_B
                    A_N(ind_azB, ind_azB) = A_BN;
                    A_N(ind_azB, ind_azB-n_az) = abs(h_now) ./ dB .* speye(n_az);
                elseif h_now > 0
                    A_N(ind_azB, ind_azB) = A_BN;
                    A_N(ind_azB, ind_azB+n_az) = abs(h_now) ./ dB .* speye(n_az); 
                else 
                    A_N(ind_azB, ind_azB) = A_BN;
                    A_N(ind_azB, ind_azB-n_az) = abs(h_now) ./ dB .* speye(n_az); 
                end
            end

            % Fill A
            ind_azBN = (i_N - 1) * n_azB + 1 : i_N * n_azB;
            varkappa_now = varkappa(:,:,:,i_N);
            varrho_now = varrho(:,:,:,i_N);
            muN_f_now = muN_f(:,:,:,i_N);
            muN_b_now = muN_b(:,:,:,i_N);
            X = spdiags(varkappa_now(:),0,n_azB,n_azB);
            P = spdiags(varrho_now(:),0,n_azB,n_azB);
            XmuN_f = spdiags(muN_f_now(:),0,n_azB,n_azB);
            XmuN_b = spdiags(muN_b_now(:),0,n_azB,n_azB);

            if i_N == 1
                A(ind_azBN,ind_azBN) = A_N + P;
                A(ind_azBN,ind_azBN+n_azB) = X + XmuN_f;
            elseif i_N == n_N
                A(ind_azBN,ind_azBN) = A_N + X;
                A(ind_azBN,ind_azBN-n_azB) = P + XmuN_b;
            else
                A(ind_azBN,ind_azBN) = A_N;
                A(ind_azBN,ind_azBN+n_azB) = X + XmuN_f;
                A(ind_azBN,ind_azBN-n_azB) = P + XmuN_b;
            end
        end

        u_v = (c(:) .^ (1 - gam) - 1 ) ./ ( 1 - gam );
        V_v = V(:);

        M = (1 ./ dt + rho) .* speye(n_azBN) - A;
        d = u_v + V_v ./ dt;

        V_new = reshape(M \ d,n_a,n_z,n_B,n_N);
        
        supnorm = max(abs(V_new(:) - V_old(:)));
        
        fprintf("Outer iter: %4d, V iter: %4d, V diff: %4.6f \n", i_iter, i_V, supnorm);
        if supnorm < tol_V
            converge = "Yes";
            break;
        end

        V = lr * V_new + (1 - lr) * V;
        V_old = V;
    end

    time_spend_V = toc(time_start_V);
    fprintf("Value function converges? %3s, number of iterations: %4.0d, time spend %5.3f s \n", converge, i_V, time_spend_V);
   
    %% KFE and Simulation
    time_start_KFE = tic;
    B_sims = zeros(n_sims,8);
    N_sims = zeros(n_sims,8);
    parfor i_sim = 1:8
        
        [B_sim,N_sim] = KFE_sim(a,B,N,B_v,N_v,B_ss,N_ss,g_ss,beta,s,par,seed_base + i_sim);
        B_sims(:,i_sim) = B_sim;
        N_sims(:,i_sim) = N_sim;

    end
    time_spend_KFE = toc(time_start_KFE);
    fprintf("Simulation time spend %5.3f s \n", time_spend_KFE);
    

    %% Update of the PLM using a neural network
    dB_sims_nn = ( B_sims(n_burnin+1:end,:) - B_sims(n_burnin:end-1,:) ) ./ dt_sim;
    dB_sims_nn = dB_sims_nn(:);
    B_sims_nn = B_sims(n_burnin:end-1,:);
    N_sims_nn = N_sims(n_burnin:end-1,:);
    B_sims_nn = B_sims_nn(:);
    N_sims_nn = N_sims_nn(:);

    Y_ave = NaN(n_Bnn, n_Nnn);

    for i_Bnn = 1:n_Bnn
        for i_Nnn = 1:n_Nnn
            Bnn_now = Bnn_v(i_Bnn);
            Nnn_now = Nnn_v(i_Nnn);
            ind_wr = abs(B_sims_nn - Bnn_now) < dBnn / 2 & ...
                abs(N_sims_nn - Nnn_now) < dNnn / 2; % index of observations within range
            if sum(ind_wr) > 5
                Y_ave(i_Bnn,i_Nnn) = mean( dB_sims_nn(ind_wr) );
            end
        end
    end


    % Standardized input and output
    Y_train = Y_ave(:);
    Y_train = Y_train(~isnan(Y_ave(:)));
    Y_w = (max(Y_train) - min(Y_train)) / 2;
    Y_m = (max(Y_train) + min(Y_train)) / 2;
    Y_train = (Y_train - Y_m) / Y_w * 4;
    B_train = Bnn(:);
    B_train = B_train(~isnan(Y_ave(:)));
    N_train = Nnn(:);
    N_train = N_train(~isnan(Y_ave(:)));
    X_train = [B_train, N_train];
    X_w = (max(X_train) - min(X_train)) / 2;
    X_m = (max(X_train) + min(X_train)) / 2;
    X_train = (X_train - X_m) ./ X_w * 4;
    
    fs_final = zeros(8,1);
    fss = zeros(n_tnn,8);
    thetas_final = zeros(length(theta),8);
    
    % training
    time_start_NN = tic;
    parfor i_try = 1:8
    
        if i_try > 1
            rng(seed_base + i_try);
            theta_init = randn(length(theta),1);
            theta_init(1) = 0;
            theta_init(n_Q+2:3:end) = 0;
        else
            theta_init = theta;
        end
        
        % line search algorithm
        [theta_train,f_small,fs] = NN_train(X_train,Y_train,theta_init,lr_theta,n_tnn,n_Q);
        
        fs_final(i_try) = f_small;
        fss(:,i_try) = fs;
        thetas_final(:,i_try) = theta_train;
    end
    
    time_spend_NN = toc(time_start_NN);
    fprintf("NN time spend %5.3f s \n", time_spend_NN);

    theta = thetas_final(:,fs_final == min(fs_final));
    
    %% Update and plot perceived law of motion
    hold off;
    h_new = Y_m + Y_w ./ 4 .* reshape(plm(([B(:),N(:)]-X_m)./X_w.*4,theta,@softplus),n_a,n_z,n_B,n_N);
    h = lr_h*h_new + (1 - lr_h)*h;
    hplot = h(1,1,:,:);
    hplot = reshape(hplot,n_B,n_N);
    surf(B_v.*ones(1,n_N),N_v'.*ones(n_B,1),hplot)
    hold on
    surf(Bnn,Nnn,Y_ave)
    title(i_iter);
    pause(1e-6);
    
    h_fine_new = Y_m + Y_w ./ 4 .* reshape(plm(([Bnn(:),Nnn(:)]-X_m)./X_w.*4,theta,@softplus),n_Bnn,n_Nnn);
    h_fine = lr_h * h_fine_new + (1 - lr_h) * h_fine_old;
    
    RMSdiff = mean( (h_fine_new(~isnan(Y_ave(:))) - h_fine_old(~isnan(Y_ave(:)))).^2 ) .^ 0.5;
    
    fprintf("Iteration: %5d, RMS diff = %4.6f \n", i_iter, RMSdiff);
    if RMSdiff < tol_V && i_iter > 5
        break;
    end
    
    h_old = h;
    theta_old = theta;
    h_fine_old = h_fine;
end

time_spend = toc(time_start);
fprintf("Total time spend: %5.3f s \n", time_spend);
save("results\results_s14z72_denseab.mat");
