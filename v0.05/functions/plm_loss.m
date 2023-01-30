function [loss] = plm_loss(X,Y,theta,phi)
%PLM Compute the perceived law of motion using NN at points X
%    X     : Points at which the function is to be evaluated
%    Y     : Training data
%    theta : NN parameters (see p. 18)
%    phi   : activation function

    
    h = plm(X,theta,phi);
    loss = mean((h - Y) .^ 2);

end

