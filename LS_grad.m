%% Gradient for Least Squares
%% f_i(x) = 1/2*norm(Ax-b)^2
function  f = LS_grad(x,A,b)
f = A'*(A*x-b);