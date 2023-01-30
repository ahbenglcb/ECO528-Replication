function [h] = plm(X,theta,phi)
%PLM Compute the perceived law of motion using NN at points X
%    X     : Points at which the function is to be evaluated
%    theta : NN parameters (see p. 18)
%    phi   : activation function

    n_obs = size(X,1);
    n_in = size(X,2);
    n_Q = ( length(theta) - 1 ) / (n_in + 2);
    
    theta20 = theta(1);
    theta2q = theta(2:n_Q+1);
    theta1 = reshape(theta(n_Q+2:end),n_in+1,n_Q);
    
    X_c = [ones(n_obs,1), X];
    
    h = theta20 + phi( X_c * theta1 )* theta2q;
end

