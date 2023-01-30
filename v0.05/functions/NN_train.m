function [theta,f_mid,fs] = NN_train(X_train,Y_train,theta_init,lr_theta,n_tnn,n_Q)
%NN_TRAIN Summary of this function goes here


    theta = theta_init;
    
    lr_theta_mid = lr_theta;
    
    fs = zeros(n_tnn,1);

    for i_theta = 1:n_tnn

        lr_theta_left = 0;
%         rglrztn = theta; % to comment
%         rglrztn(1) = 0; % to comment
%         rglrztn(n_Q+2:3:end) = 0; % to comment
        grad = plm_grad(X_train,Y_train,theta,@softplus,@softplus_grad);
%         grad = 2*(grad + 0.1 / length(Y_train) * rglrztn); % to comment

        % Current theta
        theta_left = theta;
        f_left = plm_loss(X_train,Y_train,theta,@softplus);

        % New theta with smaller loss
        lr_theta_mid = lr_theta_mid * 5;
        theta_mid = theta - lr_theta_mid .* grad;
        f_mid = plm_loss(X_train,Y_train,theta_mid,@softplus);

        while f_mid > f_left
            lr_theta_mid = lr_theta_mid / 1.5;
            theta_mid = theta - lr_theta_mid .* grad;
            f_mid = plm_loss(X_train,Y_train,theta_mid,@softplus);
        end

        % New theta with larger loss
        lr_theta_right = lr_theta_mid * 1.1;
        theta_right = theta - lr_theta_right .* grad;
        f_right = plm_loss(X_train,Y_train,theta_right,@softplus);

        while f_right < f_mid
            lr_theta_left = lr_theta_mid;
            theta_left = theta_mid;
            f_left = f_mid;
            
            lr_theta_mid = lr_theta_right;
            theta_mid = theta_right;
            f_mid = f_right;
 
            lr_theta_right = lr_theta_right * 1.1;
            theta_right = theta - lr_theta_right .* grad;
            f_right = plm_loss(X_train,Y_train,theta_right,@softplus);
        end
        
        % Find local min
        for i_lmin = 1:5
            if f_right > f_left
                lr_theta_now = (lr_theta_right + lr_theta_mid ) / 2;
                theta_now = theta - lr_theta_now .* grad;
                f_now = plm_loss(X_train,Y_train,theta_now,@softplus);

                if f_now > f_mid
                    lr_theta_right = lr_theta_now;
                    theta_right = theta_now;
                    f_right = f_now;
                else
                    lr_theta_left = lr_theta_mid;
                    theta_left = theta_mid;
                    f_left = f_mid;

                    lr_theta_mid = lr_theta_now;
                    theta_mid = theta_now;
                    f_mid = f_now;
                end
            else
                lr_theta_now = (lr_theta_left + lr_theta_mid) / 2;
                theta_now = theta - lr_theta_now .* grad;
                f_now = plm_loss(X_train, Y_train, theta_now, @softplus);

                if f_now > f_mid
                    lr_theta_left = lr_theta_now;
                    theta_left = theta_now;
                    f_left = f_now;
                else
                    lr_theta_right = lr_theta_mid;
                    theta_right = theta_mid;
                    f_right = f_mid;

                    lr_theta_mid = lr_theta_now;
                    theta_mid = theta_now;
                    f_mid = f_now;
                end
            end
        end
        
        fs(i_theta) = f_mid;
        % update theta
        theta = theta_mid;
    end
end

