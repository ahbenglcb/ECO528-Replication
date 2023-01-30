function [B_sims,N_sims] = KFE_sim(a,B,N,B_v,N_v,B_ss,N_ss,g_ss,beta,s,par,seed)
%KFE_SIM Simulation and KFE

    rng(seed);
    v2struct(par);
    alpha = par.alpha;
    dB = B_v(2) - B_v(1);
    dN = N_v(2) - N_v(1);
    da = a(2,1,1,1) - a(1,1,1,1);
    n_az = n_a * n_z;
    
    %% KFE & Simulation
    % Create a grid
    a_sim = a(:,:,1,1);

    % Create B and N grids
    B_KFE_grid = B(1,1,:,:);
    N_KFE_grid = N(1,1,:,:);

    % Simulation and KFE
    B_sims = zeros(n_sims,1);
    N_sims = zeros(n_sims,1);
    g_sims = zeros(n_a,n_z,n_sims);

    B_sims(1) = B_ss;
    N_sims(1) = N_ss;
    g_sims(:,:,1) = g_ss;

    for i_sim = 1:n_sims-1
        g_nexts = zeros(n_a,n_z,4);

        N_now = N_sims(i_sim);
        B_now = B_sims(i_sim);
        g_now = g_sims(:,:,i_sim);


        % Find nearest B and N on the grid
        % Use Euler distance
        euler_dist = (B_now - B_KFE_grid).^2 + (N_now - N_KFE_grid).^2;
        [~,ind_dist] = mink(euler_dist(:),1);

        N_near = N_KFE_grid(ind_dist);
        B_near = B_KFE_grid(ind_dist);

        ind_Nnear = find(N_v == N_near);
        ind_Bnear = find(B_v == B_near);

        if N_now - N_near > 0
            ind_NL = ind_Nnear;
        else
            ind_NL = ind_Nnear - 1;
        end

        if B_now - B_near > 0
            ind_BL = ind_Bnear;
        else
            ind_BL = ind_Bnear - 1;
        end


        wB = (B_v(ind_BL + 1) - B_now) ./ dB;
        wN = (N_v(ind_NL + 1) - N_now) ./ dN;

        for i_near = 1:4

            if i_near == 1
                i_Nnear = ind_NL;
                i_Bnear = ind_BL;
            elseif i_near == 2
                i_Nnear = ind_NL + 1;
                i_Bnear = ind_BL;
            elseif i_near == 3
                i_Nnear = ind_NL;
                i_Bnear = ind_BL + 1;
            elseif i_near == 4
                i_Nnear = ind_NL + 1;
                i_Bnear = ind_BL + 1;
            end




            % Construct matrix A and compute next period g
            beta_now = beta(:,:,i_Bnear,i_Nnear);
            s_now = s(:,:,i_Bnear,i_Nnear);
            A_BN = spdiags(beta_now(:),0,n_az,n_az) ...
                + spdiags(max(s_now(:) ./ da, 0),-1,n_az,n_az)' ...
                + spdiags(max(- s_now(:) ./ da, 0),1,n_az,n_az)' ...
                + spdiags(lambda1*ones(n_a,1),-n_a,n_az,n_az)' ...
                + spdiags(lambda2*ones(n_a,1),-n_a,n_az,n_az);

            g_nexts(:,:,i_near) = reshape(( speye(n_az) - dt_sim .* A_BN' ) \ g_now(:), n_a, n_z);
            g_nexts(:,:,i_near) = g_nexts(:,:,i_near) ./ sum(g_nexts(:,:,i_near)*da,'all');

        %     hold off
        %     plot(g_now(:,1));
        %     hold on
        %     plot(g_now(:,2));
        %     pause(1e-6);


        end

        g_next = wB * wN .* g_nexts(:,:,1) ...
            + wB * (1 - wN) .* g_nexts(:,:,2) ...
            + (1 - wB) * wN .* g_nexts(:,:,3) ...
            + (1 - wB) * (1 - wN) .* g_nexts(:,:,4);

        % Compute next period B
        B_sims(i_sim + 1) = a_sim(:)' * g_next(:) * da;
        if B_sims(i_sim + 1) <= B_min
            B_sims(i_sim + 1) = B_min + 1e-5;
        elseif B_sims(i_sim + 1) >= B_max
            B_sims(i_sim + 1) = B_max - 1e-5;
        end

        % Compute next period N
        r_now = alpha * (B_now + N_now) ^ (alpha - 1) - delta ...
            - sig^2 * (B_now + N_now) / N_now;
        N_sims(i_sim + 1) = N_now + ...
            ( alpha * (B_now + N_now) ^ alpha - delta * (B_now + N_now) ...
            - r_now * B_now - rhohat * N_now ) * dt_sim ...
            + sig * (B_now + N_now) * dt_sim ^ 0.5 * normrnd(0,1) ;
        if N_sims(i_sim + 1) <= N_min
            N_sims(i_sim + 1) = N_min + 1e-5;
        elseif N_sims(i_sim + 1) >= N_max
            N_sims(i_sim + 1) = N_max - 1e-5;
        end

        g_sims(:,:,i_sim+1) = g_next;
    end

end

