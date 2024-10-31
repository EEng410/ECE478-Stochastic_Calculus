clc
clear 
close all
%%

A = [4/3, 14/29; 7/3, 9/29; 11/3 -39/29]
b = [2, 3, 3]

A \ b' 


%% Exact Simulation

function V = compute_V(S_N)
end

function E = compute_E_p_V(V_N_tilde, N, r, d, u)
    % V_N = (1+r)^N * V_N_tilde;
    %S_N = d^N (u/d)^phi_n, phi_n number of heads
    ind = 0:N;
    S = d^ind .* (u/d)^ind;
    comb_vec = factorial(N)./factorial(N-ind)./factorial(ind);
    E_vec = compute_V(S).*(1./(1+r)^ind).* comb_vec.*p^ind .* (1-p)^(N-ind);
    E = sum(E_vec);
end

function [X_next, delta_next] = replicate_step(X_n, delta_n)
    
end