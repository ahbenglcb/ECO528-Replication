function [phi] = softplus(x)
%SOFTPLUS Softplus activation function

    phi = log( 1 + exp(x) );
end

