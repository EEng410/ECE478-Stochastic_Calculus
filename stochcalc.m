clc
clear
close all
%% 

delta = 1/260;  % one day
N = 520;        % two years

T = delta*N;

S_0 = 1;
alpha = 0.1;
sigma_vec = [0.05, 0.1, 0.3];
r = 0.05;

% Choose a sigma value
sigma = sigma_vec(1);

num_paths = 1000;
% paths = reshape([paths_1; paths_2; paths_3], [3, num_paths, N+1]);
paths = zeros(3, num_paths, N+1);

% Generate three sets of paths
paths(1, :, :) = genPaths(alpha, sigma_vec(1), r, delta, num_paths, S_0, N);
paths(2, :, :) = genPaths(alpha, sigma_vec(2), r, delta, num_paths, S_0, N);
paths(3, :, :) = genPaths(alpha, sigma_vec(3), r, delta, num_paths, S_0, N);




% Generate three sets of E[S[N]] and E[S[N/2]]
E_S_N_over_2 = mean(paths(:, :, N/2), 2)'
E_S_N = mean(paths(:, :, N), 2)'

% Theoretical values:

% random_val = randn(1);
random_val = 0; % On average, this random value will be 0...?
S_N_over_2_theoretical = S_0 * exp(sigma_vec.*(sqrt(N/2*delta)*random_val+(alpha-r)./sigma_vec.*(N/2)*delta)+(r-sigma.^2/2)*N/2*delta)
S_N_theoretical = S_0 * exp(sigma_vec.*(sqrt(N*delta)*random_val+(alpha-r)./sigma_vec.*(N)*delta)+(r-sigma_vec.^2/2)*N*delta)

%% Use Black Scholes to derive a formula for V:
% V = N(d+)S_t - N(d-)Ke^(-rT-t)
t = N/2*delta;

S_t = linspace(0.5, 1.5, 1000);

[~, V_1] = V_BSM(S_t, alpha, sigma_vec(1), r, T, t, S_0);
[~, V_2] = V_BSM(S_t, alpha, sigma_vec(2), r, T, t, S_0);
[~, V_3] = V_BSM(S_t, alpha, sigma_vec(3), r, T, t, S_0);

figure
plot(S_t, V_1)
hold on
plot(S_t, V_2)
plot(S_t, V_3)
hold off
xlabel('S[N/2]')
ylabel('V[N/2]')
title('V vs S at time N/2*delta')
legend('\sigma = 0.05', '\sigma = 0.1', '\sigma = 0.3')

%% Comparing V

% Grab first 10 paths for sigma(1)
sig_index = 2;
paths_new = paths(sig_index, 1:10, :);

% Compute V from BSM:

N_start = N/2+1;
S_start = (paths_new(:, :, N_start));
t = N_start*delta;
[~, V] = V_BSM(S_start, alpha, sigma_vec(sig_index), r, T, t, S_0);
V_real = squeeze(V);
V_est = zeros(10, 1);
for i = 1:10
    [paths_S, paths_V] = genPaths_V(alpha, sigma_vec(sig_index), r, delta, 1000, S_start(i), N_start, N);
    V_est(i) = mean(paths_V(1:end, end));
end

figure
scatter(S_start, V_est)
hold on
scatter(S_start, V_real)
hold off
xlabel("S^{(i)}[N/2]")
ylabel("V^{(i)}[N/2]")
title("Comparing Estimated V (MC) and Actual V (BSM)")

%% Jump Processes

% Modify my genPaths function with two types of jump processs - one with
% constant jump height and one with random jump height

p = 0.01;
jump_height = 0.1;
jump_process = genPathsSimpleJumps(alpha, sigma_vec(1), r, delta, 1, S_0, N, p, jump_height);

figure
time = delta*(1:521);
plot(time, jump_process(1, :))
xlabel("Time")
ylabel("Path")
title("A Simple Poisson Process")

%% Compound Jump Process
jump_height_var = 0.005;
jump_process = genPathsCompoundJumps(alpha, sigma_vec(1), r, delta, 1, S_0, N, p, jump_height_var);

figure
time = delta*(1:521);
plot(time, jump_process(1, :))
xlabel("Time")
ylabel("Path")
title("A Compound Poisson Process")
%% Functions
% Implement SDE
function paths = genPaths(alpha, sigma, r, delta, num_paths, S_0, N)
    % generate a path - allocate space 
    paths = zeros(num_paths, N+1); % Actually N+1 total points
    paths(:, 1) = S_0;  % Initialize path
    path = 1;
    while path <= num_paths
        % Modified SDE: dS = r S dt + sigma S dW
        for step = 1:N
            S = paths(path, step);
            dW = sqrt(delta) * randn(1, 1);
            dW_tilde = (alpha - r)/sigma*delta + dW;
            dS = r * S * delta + sigma * S * dW_tilde;
            paths(path, step + 1) = S + dS;
            if paths(path, step+1) < 0
                disp("Negative Path Detected. Pruning Path.")
                endloop;
            end
        end
        path = path + 1;
    end
end

function [paths_S, paths_V] = genPaths_V(alpha, sigma, r, delta, num_paths, S_start, N_start, N)
    % generate a path - allocate space 
    paths_S = zeros(num_paths, N-N_start+1); % Actually N+1 total points
    paths_V = zeros(num_paths, N-N_start+1); % Actually N+1 total points
    K = exp(alpha*N*delta);
    
    paths_S(:, 1) = S_start;
    paths_V(:, 1) = computeV(S_start, K);  % Initialize path
    path = 1;
    while path <= num_paths
        % Modified SDE: dS = r S dt + sigma S dW_tilde
        for step = 1:N-N_start
            S = paths_S(path, step);
            dW = sqrt(delta) * randn(1, 1);
            % dW_tilde = (alpha - r)/sigma*delta + dW;
            dW_tilde = dW;
            dS = r * S * delta + sigma * S * dW_tilde;
            paths_S(path, step + 1) = S + dS;
            if paths_S(path, step+1) < 0
                disp("Negative Path Detected. Pruning Path.")
                endloop;
            end
            % K = exp(alpha*(N_start+step)*delta);

            paths_V(path, step + 1) = computeV(S+dS, K);
        end
        path = path + 1;
    end
end
% Compute V from BSM
function [S_t, V] = V_BSM(S_t, alpha, sigma, r, T, t, S_0)
    K = exp(alpha*T)*S_0;
    d_p = 1./(sigma.*sqrt(T-t)).*(log(S_t./K)+(r+(sigma.^2)/2)*(T-t));
    % d_m = d_p - sigma.*sqrt(T-t);
    d_m = 1./(sigma.*sqrt(T-t)).*(log(S_t./K)+(r-(sigma.^2)/2)*(T-t));
    V = normcdf(d_p).* S_t - normcdf(d_m).*K.*exp(-r*(T-t));
end

function V = computeV(S, K)
    V = max(S - K, 0);
end

function paths = genPathsSimpleJumps(alpha, sigma, r, delta, num_paths, S_0, N, p, jump_height)
    % generate a path - allocate space 
    paths = zeros(num_paths, N+1); % Actually N+1 total points
    paths(:, 1) = S_0;  % Initialize path
    path = 1;
    while path <= num_paths
        % Modified SDE: dS = r S dt + sigma S dW
        for step = 1:N
            S = paths(path, step);
            dW = sqrt(delta) * randn(1, 1);
            dW_tilde = (alpha - r)/sigma*delta + dW;
            dS = r * S * delta + sigma * S * dW_tilde;
            
            % Simple Jump Process
            jump_test = rand(1, 1);
            jump = jump_height * (jump_test < p); 

            paths(path, step + 1) = S + dS + jump;

            if paths(path, step+1) < 0
                disp("Negative Path Detected. Pruning Path.")
                endloop;
            end
        end
        path = path + 1;
    end
end

function paths = genPathsCompoundJumps(alpha, sigma, r, delta, num_paths, S_0, N, p, jump_height_var)
    % generate a path - allocate space 
    paths = zeros(num_paths, N+1); % Actually N+1 total points
    paths(:, 1) = S_0;  % Initialize path
    path = 1;
    while path <= num_paths
        % Modified SDE: dS = r S dt + sigma S dW
        for step = 1:N
            S = paths(path, step);
            dW = sqrt(delta) * randn(1, 1);
            dW_tilde = (alpha - r)/sigma*delta + dW;
            dS = r * S * delta + sigma * S * dW_tilde;
            
            % Simple Jump Process
            jump_height = sqrt(jump_height_var)*randn(1, 1);
            jump_test = rand(1, 1);
            jump = jump_height * (jump_test < p); 

            paths(path, step + 1) = S + dS + jump;

            if paths(path, step+1) < 0
                disp("Negative Path Detected. Pruning Path.")
                endloop;
            end
        end
        path = path + 1;
    end
end