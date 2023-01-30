function [grad] = plm_grad(X,Y,theta,phi,phi_grad)
%PLM_GRAD Compute the gradient of the perceived law of motion using NN 
%    at points X
%    X     : Points at which the function is to be evaluated
%    theta : NN parameters (see p. 18)
%    phi   : activation function
%    Y     : Training data


    n_obs = size(X,1);
    n_in = size(X,2);
    n_Q = ( length(theta) - 1 ) / (n_in + 2);
    
    theta20 = theta(1);
    theta2q = theta(2:n_Q+1);
    theta1 = reshape(theta(n_Q+2:end),n_in+1,n_Q);
    
    X_c = [ones(n_obs,1), X];

    h = plm(X,theta,phi);
    
    grad_20 = mean( h - Y );
    grad_2q = mean( (h - Y) .* phi( X_c * theta1 ) );
    grad_1 = mean( theta2q' .* (h - Y) .* phi_grad( X_c * theta1 ) );
    
    for i_in = 1:n_in
        grad_1 = [grad_1; mean( X(:,i_in) .* theta2q' .* (h - Y) .* phi_grad( X_c * theta1 ) )];
    end
    
    grad = [grad_20; grad_2q(:); grad_1(:)];
end

