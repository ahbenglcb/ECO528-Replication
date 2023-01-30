function [phi_grad] = softplus_grad(x)
%SOFTPLUS_GRAD Derivative of softplus activation function

    phi_grad = 1 ./ ( 1 + exp(-x) );
end