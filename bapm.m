clc
clear 
close all
%% Exact Simulation

r = 0.05;
u = 1.1;
d = 1.01;
N = 5;

p = (1+r-d)/(u-d);
[X, delta] = cringe(N, r, d, u);

% for i = 3
%     paths = decimalToBinaryVector(0:2^(i-1)-1);
%     ind = sum((paths == 0), 2);
%     not_ind = sum((paths == 1), 2);
%     comb_vec = p.^ind.*(1-p).^not_ind;
%     disp([i, dot(cell2mat(X(i, 1:2^(i-1))), comb_vec)])
% end

function V = compute_V(S_N, r,  N)
    K = (1+r)^N;
    V = (S_N - K) * ((S_N - K) > 0); 
end

function S = compute_S(bin_arr, r, d, u)
    % Assume S_0 = 1
    % Assume bin_arr is a binary vector of length N, indicating path length
    N = size(bin_arr, 2);
    phi = sum(bin_arr == 1);
    S = u^phi*d^(N-phi);
end

function E = compute_E_p_V(N, r, d, u)
    % V_N = (1+r)^N * V_N_tilde;
    %S_N = d^N (u/d)^phi_n, phi_n number of heads
    ind = 0:N;
    S = d^ind .* (u/d)^ind;
    comb_vec = factorial(N)./factorial(N-ind)./factorial(ind);
    E_vec = compute_V(S).*(1./(1+r)^ind).* comb_vec.*p^ind .* (1-p)^(N-ind);
    E = sum(E_vec);
end

function [X_n, delta_n] = replicate_step(S_n, r, d, u, N)
    % [compute_V(u*S_n, r, N+1) ; compute_V(d*S_n, r, N+1)] 
    % [1 - r, (u - (1-r))*S_n ; 1 - r, (d - (1-r))*S_n]
    out = [compute_V(u*S_n, r, N+1) ; compute_V(d*S_n, r, N+1)] \ [1 - r, (u - (1-r))*S_n ; 1 - r, (d - (1-r))*S_n];
    X_n = out(1);
    delta_n = out(2);
end

function [X, delta] = cringe(N, r, d, u)
    delta = cell(N, 1);
    X = cell(N, 1);
    for ind = N:-1:1
        n = ind-1;
        % Assume path dependence
        X{ind} = zeros(2^n, 1);
        delta{ind} = zeros(2^n, 1);
        % Compute all paths to time step N
        paths = decimalToBinaryVector(0:2^n-1);
        for path = 1:size(paths, 1)
            [xtemp, temp] = replicate_step(compute_S(paths(path, :), r, d, u), r, d, u, n); 
            X{ind, path} = xtemp;
            delta{ind, path} = temp;
        end
    end
end