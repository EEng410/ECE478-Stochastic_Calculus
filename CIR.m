clc
clear
close all
%% CIR Model 

beta = 1;
alpha = 0.10*beta;
r = 0.05;
delta = 0.01;
T = 10;
sigma = 0.5;
R_0 = r;
num_paths = 1000;
N = T/delta;

[paths, invalid_paths_ind] = genPathsR(alpha, sigma, beta, delta, num_paths, R_0, N);
disp(sum(invalid_paths_ind))

% Get first 10 valid paths:

valid_paths = paths(-1*(invalid_paths_ind-1)==1, :);
invalid_paths = paths(invalid_paths_ind == 1, :);
sample_means = mean(valid_paths(:, 101:100:end))
sample_variance = mean(valid_paths(:, 101:100:end).^2 - sample_means.^2)
% This plot is completely unintelligible
figure
plot(valid_paths(1, :))
hold on
for i = 2:10
    plot(valid_paths(i, :))
end
plot(invalid_paths(1, :))
plot(invalid_paths(2, :))
hold off

%% Using X = log R
X_0 = log(R_0);
X_paths = genPathsX(alpha, sigma, beta, delta, num_paths, X_0, N);

R_paths = exp(X_paths);

figure
plot(R_paths(1, :))
hold on
for i = 2:10
    plot(R_paths(i, :))
end

valid_paths = not(isnan(R_paths(:, end)) | any(R_paths > 10, 2));

R_paths_valid = R_paths(valid_paths, 101:100:end);

sample_means_new = mean(R_paths(valid_paths, 101:100:end))
% sample_variance_new = mean(R_paths(valid_paths, 101:100:end).^2 - sample_means_new.^2)

sample_variance_new = var(R_paths(valid_paths, 101:100:end))

%% Plots
t = 1:10;
E_R = exp(-beta*t)*r+alpha/beta*(1-exp(-beta*t))
var_R = sigma^2/beta*r *(exp(-beta*t)-exp(-2*beta*t))+alpha*sigma^2/(2*beta^2)*(1-2*exp(-beta*t)-exp(-2*beta*t))
figure
scatter(t, sample_means)
hold on
scatter(t, sample_means_new)
yline(alpha/beta)
plot(t, E_R)
hold off

figure
scatter(t, sample_variance)
hold on
scatter(t, sample_variance_new)
yline(alpha*sigma^2/2/beta^2)
plot(t, var_R)
hold off


%% Functions 

function [paths, invalid_paths] = genPathsR(alpha, sigma, beta, delta, num_paths, R_0, N)
    % generate a path - allocate space 
    paths = zeros(num_paths, N+1); % Actually N+1 total points
    paths(:, 1) = R_0;  % Initialize path
    invalid_paths = zeros(num_paths, 1);
    for path = 1:num_paths
        % SDE: dR = (alpha - beta * R)* dt + sigma * sqrt(R) * dW
        for step = 1:N
            R = paths(path, step);
            dW = sqrt(delta) * randn(1, 1);
            dR = (alpha - beta*R) * delta + sigma * sqrt(R) * dW;
            paths(path, step + 1) = R + dR;
            if paths(path, step+1) < 0
                % Terminate negative path
                disp("Negative Path Detected. Terminating Path.")
                invalid_paths(path) = 1;
                break;
            end
        end
    end
end


function paths = genPathsX(alpha, sigma, beta, delta, num_paths, X_0, N)
    % generate a path - allocate space 
    paths = zeros(num_paths, N+1); % Actually N+1 total points
    paths(:, 1) = X_0;  % Initialize path
    for path = 1:N
        % SDE: dR = (alpha - beta * R)* dt + sigma * sqrt(R) * dW
        % dX = dR/R - 1/(2R^2)dRDR
        for step = 1:N
            X = paths(path, step);
            dW = sqrt(delta) * randn(1, 1);
            % dX = (alpha/exp(X) - beta) * delta + sigma *(exp(-0.5*X)) * dW - 1/(2*exp(X)^2)*(sigma*sqrt(exp(X)))^2*delta;
            R = exp(X);
            dX = (alpha/R - beta - sigma^2/(2*R))* delta + sigma /sqrt(R)* dW;
            paths(path, step + 1) = X + dX;
        end
    end
end