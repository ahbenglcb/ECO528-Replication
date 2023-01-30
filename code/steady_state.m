%
% To compute steady state
%

%% Create steady state grid
a_ss = a(:,:,1,1);
z_ss = z(:,:,1,1);
lambda_ss = lambda(:,:,1,1);

%% Compute steady state object
K_ss = ( ( delta + rhohat ) / alpha ) ^ (1/(alpha - 1));
r_ss = rhohat;
rc_ss = alpha * K_ss ^ (alpha - 1);
w_ss = (1 - alpha) * K_ss ^ alpha;

%% Steady state value function V
V_ss = (w_ss .* z_ss + r_ss .* a_ss) .^ (1 - gam) / (1 - gam) / rho;
V_ss_old = V_ss;

converge = "No";
for i_V = 1:1000
    dV_ss_af = zeros(n_a, n_z); dV_ss_ab = zeros(n_a, n_z);

    % Derivatives
    dV_ss_af(1:end-1,:) = ( V_ss(2:end,:) - V_ss(1:end-1,:) ) ./ da;
    dV_ss_af(end,:) = (w_ss .* z(end,:,1,1) ...
        + r_ss .* a(end,:,1,1) ) .^ (- gam) ;
    dV_ss_ab(2:end,:) = ( V_ss(2:end,:) - V_ss(1:end-1,:) ) ./ da;
    dV_ss_ab(1,:) = (w_ss .* z(1,:,1,1) ...
        + r_ss .* a(1,:,1,1) ) .^ (- gam) ;

    % Consumption, forward
    c_ss_f = dV_ss_af .^ (- 1 / gam);
    s_ss_f = w_ss .* z_ss + r_ss .* a_ss - c_ss_f;

    % Consumption, backward
    c_ss_b = dV_ss_ab .^ (- 1 / gam);
    s_ss_b = w_ss .* z_ss + r_ss .* a_ss - c_ss_b;

    % Consumption, undetermined sign
    c_ss_u = w_ss .* z_ss + r_ss .* a_ss;
    s_ss_u = zeros(n_a,n_z);
    dV_ss_au = c_ss_u .^ (- gam);

    ind_ss_f = s_ss_f > 0;
    ind_ss_b = s_ss_b < 0;
    ind_ss_u = 1 - ind_ss_f - ind_ss_b;

    dV_ss_a = ind_ss_f .* dV_ss_af + ind_ss_b .* dV_ss_ab + ind_ss_u .* dV_ss_au;
    c_ss = dV_ss_a .^ (-1 / gam);
    s_ss = w_ss .* z_ss + r_ss .* a_ss - c_ss;

    % Create matrix A;
    n_az = n_a * n_z;
    A_ss = spdiags(ind_ss_f(:) .* s_ss(:) ./ da, -1, n_az, n_az)' ...
        - spdiags(ind_ss_b(:) .* s_ss(:) ./ da, 1, n_az, n_az)' ...
        + spdiags(ind_ss_b(:) .* s_ss(:) ./ da - ind_ss_f(:) .* s_ss(:) ./ da ...
        - lambda_ss(:), 0, n_az, n_az) ...
        + spdiags(lambda_ss(:), -n_a, n_az, n_az)' ...
        + spdiags(lambda_ss(:), n_a, n_az, n_az)';

    % Create u
    u_ss = ( c_ss .^ (1 - gam) - 1 ) ./ (1 - gam);

    % Create M and d
    M_ss = (1 / dt + rho) .* speye(n_az, n_az) - A_ss;
    d_ss = u_ss(:) + 1 ./ dt .* V_ss(:);

    % Update V
    V_ss_new = reshape( M_ss \ d_ss, n_a, n_z);

    if max(abs(V_ss_new(:) - V_ss_old(:))) < tol_V
        converge = "Yes";
        break;
    end

    V_ss = lr * V_ss_new + (1 - lr) * V_ss;
    V_ss_old = V_ss;

end

fprintf("Steady state value function converges? %3s, number of iterations: %4.0d \n", converge, i_V);

%% Steady state distribution g
g_ss = ones(n_a, n_z) ./ (n_az * da) ;
g_ss_old = g_ss;

converge = "No";
for i_g = 1:1000
    g_ss_new = reshape((speye(n_az) - dt .* A_ss') \ g_ss(:), n_a, n_z);
    if max(abs(g_ss_new(:) - g_ss(:))) < tol_V
        converge = "Yes";
        break;
    end
    g_ss = lr .* g_ss_new + (1 - lr) * g_ss;
    g_ss_old = g_ss;
end
fprintf("Steady state distribution converges? %3s, number of iterations: %4.0d \n", converge, i_g);

%% Steady state B and N
B_ss = a_ss(:)' * g_ss(:) * da;
N_ss = K_ss - B_ss;






