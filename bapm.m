clc
clear 
close all
%% Exact Simulation

r = 0.05;
u = 1.1;
d = 1.01;
N = 5;
S_0 = 1;

p = (1+r-d)/(u-d);
[X, delta, S_tilde, V_tilde] = cringe(N, r, d, u);

E_S_p1 = compute_E_S_tilde(N-1, S_tilde, p-0.05) 
E_V_p1 = compute_E_S_tilde(N-1, V_tilde, p-0.05) 
% the function to compute S_tilde can be used for V as well. The result is 
% that we get 0.0212 in the expectation just before the first time step, 
% which is exactly the same as X_0 computed by our simulation!

E_S_p2 = compute_E_S_tilde(N-1, S_tilde, p+0.05) 
E_V_p2 = compute_E_S_tilde(N-1, V_tilde, p+0.05) 

E_S = compute_E_S_tilde(N-1, S_tilde, p)
E_V = compute_E_S_tilde(N-1, V_tilde, p) 

X_0 = X{1, 1}

%% Monte Carlo
N = 5;
M = 32;
w_n = [];
[S_0_est, V_0_est] = monte_Carlo(w_n, M, N, p, r, u, d)

% The estimates get more refined as M gets higher. Generally, the S
% converges to 1 and the V converges to the neighborhood of the right
% value. 

%% Different Sim

N = 100;
r = 10^-3;
u = 1 + 5 *10^-3;
d = 1 + 10^-4;
M = 10000;

p = (1+r-d)/(u-d);

[S_0_est, V_0_est] = monte_Carlo(w_n, M, N, p, r, u, d)

% As we increase M, S_0 still stays at 1, which is good. The V_0 value
% seems to converge to 0.0080 and the prediction becomes much more stable.
% The prediction is quite fast even for M = 100000.

%% No Longer Estimating S_0

% Generate random paths
p_new = 0.9*p;
new_paths = rand(5, 10) < p_new;
S_10 = u.^sum(new_paths == 1, 2) .* d.^sum(new_paths == 0, 2);



M = 10000;

S_10_est = zeros(5, 1);
V_10_est = zeros(5, 1);

% The following return S_tilde and V_tilde values
[S_10_est(1), V_0_est(1)] = monte_Carlo(new_paths(1, :), M, N, p_new, r, u, d);
[S_10_est(2), V_0_est(2)] = monte_Carlo(new_paths(2, :), M, N, p_new, r, u, d);
[S_10_est(3), V_0_est(3)] = monte_Carlo(new_paths(3, :), M, N, p_new, r, u, d);
[S_10_est(4), V_0_est(4)] = monte_Carlo(new_paths(4, :), M, N, p_new, r, u, d);
[S_10_est(5), V_0_est(5)] = monte_Carlo(new_paths(5, :), M, N, p_new, r, u, d);

S_10_est = S_10_est * (1+r)^10;
V_10_est = V_10_est * (1+r)^10;

S_err = S_10 - S_10_est

%% Functions
function [S_n, V_n] = monte_Carlo(w_n, M, m, p, r, u, d)
    % Y_n a conditional expectation
    % w_n a prescribed path
    % M is the number of repetitions
    % m is the number of additional time steps

    % Use the risk neutral measure
    X_start = sum(w_n == 1);
    N = numel(w_n);

    new_paths = rand(m, M) < p;
   
    X_paths = sum(new_paths, 1)+X_start;
    S = u.^sum(new_paths == 1, 1) .* d.^sum(new_paths == 0, 1);
    S_tilde = S / (1+r)^(N+m);
    S_n = mean(S_tilde, 2);

    V = compute_V(S, r,  N+m);
    V_n = mean(V, 2) / (1+r)^(N+m);
end

function E = compute_E_S_tilde(N_steps, S_tilde, p)
    E = zeros(N_steps, 1);
    E(1) = cell2mat(S_tilde(1, 1));
    for i = 1:N_steps
        ind = 0:i;
        comb_vec = factorial(i)./factorial(i-ind)./factorial(ind).* p.^ind.*(1-p).^(i-ind);
        E(i+1) = dot(comb_vec, cell2mat(S_tilde(i+1, 1:i+1)));
    end
end

% European call option
function V = compute_V(S_N, r,  N)
    K = (1+r).^N;
    V = (S_N - K) .* ((S_N - K) >= 0); 
end

% Artifact of checking path indep/dependence
% function S = compute_S(bin_arr, r, d, u)
%     % Assume S_0 = 1
%     % Assume bin_arr is a binary vector of length N, indicating path length
%     N = size(bin_arr, 2);
%     phi = sum(bin_arr == 1);
%     S = u^phi*d^(N-phi);
% end

% I did not anticipate using cells when I wrote this, so this is also an
% artifact.
% function E = compute_E_p_V(N, r, d, u, p)
%     % V_N = (1+r)^N * V_N_tilde;
%     %S_N = d^N (u/d)^phi_n, phi_n number of heads
%     ind = 0:N;
%     S = d^ind .* (u/d).^ind;
%     comb_vec = factorial(N)./factorial(N-ind)./factorial(ind);
%     E_vec = compute_V(S).*(1./(1+r)^ind).* comb_vec.*p^ind .* (1-p)^(N-ind);
%     E = sum(E_vec);
% end

function [X_n, delta_n] = replicate_step(S_n, r, d, u, N)
    % [compute_V(u*S_n, r, N+1) ; compute_V(d*S_n, r, N+1)] 
    % [1 - r, (u - (1-r))*S_n ; 1 - r, (d - (1-r))*S_n]
    A = [1 + r, (u - (1+r))*S_n ; 1 + r, (d - (1+r))*S_n];
    out = A^-1*[compute_V(u*S_n, r, N+1) ; compute_V(d*S_n, r, N+1)];
    X_n = out(1);
    delta_n = out(2);
end

function [X, delta, S, V] = cringe(N, r, d, u)
    delta = cell(N, 1);
    X = cell(N, 1);
    S = cell(N, 1);
    V = cell(N, 1);
    for ind = N:-1:1
        n = ind-1;
        % Assume path dependence
        % X{ind} = zeros(2^n, 1);
        % delta{ind} = zeros(2^n, 1);
        % Compute all paths to time step N
        % paths = decimalToBinaryVector(0:2^n-1);
        % for path = 1:size(paths, 1)
        %     [xtemp, temp] = replicate_step(compute_S(paths(path, :), r, d, u), r, d, u, n); 
        %     X{ind, path} = xtemp;
        %     delta{ind, path} = temp;
        % end
        % The experiment above proved path independence of delta. Hence,
        % only look at N+1 values per time step
        X{ind} = zeros(n+1, 1);
        delta{ind} = zeros(n+1, 1);
        S{ind} = zeros(n+1, 1);
        for heads = 0:n
            % Compute an S vector for each state
            S_val = u^heads*d^(n-heads);
            V_val = compute_V(S_val, r, n);
            S{ind, heads+1} = S_val/(1+r)^n;
            V{ind, heads+1} = V_val/(1+r)^n;
            [xtemp, temp] = replicate_step(S_val, r, d, u, n); 
            X{ind, heads+1} = xtemp;
            delta{ind, heads+1} = temp;
        end
    end
end