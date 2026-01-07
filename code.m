% This is the adapted MATLAB code for the SAO-P2PS algorithm with comparisons to
% Random Peer Selection, Nearest-Neighbor Selection (NNS), and PSO for FANETs peer selection.
% It includes:
% - main.m: The main script to set up the FANET scenario, run SAO, PSO, NNS, and Random, and compare results.
% - SAO.m: The fixed version of the SAO algorithm (corrected for bugs in the original).
% - initialization.m: Unchanged from the original.
% - p2p_cost.m: The custom cost function for the peer selection problem.
% - PSO.m: A new PSO implementation for comparison.
% Save each as separate files and run main.m.
clc;
clear;

% FANET parameters
N = 10; % Number of UAVs
dim = N;
lb = zeros(1, dim);
ub = ones(1, dim);
space_size = 50; % Reduced for more valid links
max_range = 60; % Increased
th_energy = 0.2;
P = 1;
noise = 0.01;
B = 1;
alpha = 2;
M = 10;

% Algorithm parameters
nAgents = 50; % Number of agents/particles
Max_Iter = 100; % Maximum number of iterations

% Generate UAV positions and distances
pos = space_size * rand(N, 3);
dist = pdist2(pos, pos);
energies = 0.5 + 0.5 * rand(N, 1);

disp('1. UAV movement initialization / peer initialization');
disp('UAV Positions:');
disp(pos);

% Cost function with default sigma
sigma = 0.1;
fobj = @(x) p2p_cost(x, dist, energies, max_range, th_energy, P, noise, sigma, B, alpha, M);

% Run SAO
[SAO_Fit, SAO_Pos, SAO_Convergence] = SAO(nAgents, Max_Iter, lb, ub, dim, fobj);

% Run PSO
[PSO_Fit, PSO_Pos, PSO_Convergence] = PSO(nAgents, Max_Iter, lb, ub, dim, fobj);

% Nearest-Neighbor Selection (NNS)
NNS_Pos = zeros(1, N);
for i = 1:N
    distances = dist(i,:);
    distances(i) = inf; % Exclude self
    [~, idx] = min(distances);
    NNS_Pos(i) = (idx - 1) / (N - 1); % Map to [0,1]
end
NNS_Fit = fobj(NNS_Pos);

% Random Peer Selection
Random_Pos = rand(1, N);
Random_Fit = fobj(Random_Pos);

% Compute metrics for each method with default sigma
methods = {'SAO', 'PSO', 'NNS', 'Random'};
fits = [SAO_Fit, PSO_Fit, NNS_Fit, Random_Fit];
positions = {SAO_Pos, PSO_Pos, NNS_Pos, Random_Pos};
results = zeros(4, 4); % Throughput, Fairness, EE, Utility

for m = 1:length(methods)
    x = positions{m};
    a = sort(x);
    [~, a] = sort(x);
    rates = zeros(N,1);
    dists_link = zeros(N,1);
    for i = 1:N
        j = a(i);
        if j >=1 && j <=N && dist(i,j) <= max_range && j ~= i && energies(i) >= th_energy && energies(j) >= th_energy
            d = dist(i,j);
            dists_link(i) = d;
            lq = 1 / d^alpha;
            thr = 0;
            for k = 1:M
                lq_noisy = lq * (1 + sigma * randn());
                if lq_noisy < 0
                    lq_noisy = 0;
                end
                snr = P * lq_noisy / noise;
                thr = thr + B * log2(1 + snr);
            end
            rates(i) = thr / M;
        end
    end
    throughput = sum(rates);
    non_zero_rates = rates(rates > 0);
    if isempty(non_zero_rates)
        fairness = 1;
    else
        fairness = (sum(non_zero_rates)^2) / (length(non_zero_rates) * sum(non_zero_rates.^2));
    end
    valid_count = length(non_zero_rates);
    if valid_count > 0
        energy_cons = sum(dists_link(rates > 0).^2) + valid_count * P;
    else
        energy_cons = 1;
    end
    ee = throughput / energy_cons;
    utility = -fits(m); % Negative because cost is minimized
    results(m,:) = [throughput, fairness, ee, utility];
end

% Display results for default sigma
disp('Comparison of Methods for sigma=0.1:');
disp('Method | Throughput | Fairness | Energy Efficiency | Utility');
for m = 1:length(methods)
    fprintf('%s | %.4f | %.4f | %.4f | %.4f\n', methods{m}, results(m,1), results(m,2), results(m,3), results(m,4));
end

% Plot convergence
figure('Position', [500 500 800 400])
plot(1:Max_Iter, -SAO_Convergence, 'r-', 'LineWidth', 2, 'DisplayName', 'SAO-P2PS');
hold on;
plot(1:Max_Iter, -PSO_Convergence, 'b--', 'LineWidth', 2, 'DisplayName', 'PSO');
plot(1:Max_Iter, ones(1, Max_Iter) * -NNS_Fit, 'g:', 'LineWidth', 2, 'DisplayName', 'NNS');
plot(1:Max_Iter, ones(1, Max_Iter) * -Random_Fit, 'k-.', 'LineWidth', 2, 'DisplayName', 'Random');
title('Convergence Comparison')
xlabel('Iteration')
ylabel('Utility')
legend('show')
grid on
box on

% Bar plot for metrics
figure('Position', [500 500 800 400])
subplot(1,4,1)
bar(results(:,1))
set(gca, 'XTickLabel', methods)
title('Throughput')
ylabel('bps/Hz')
subplot(1,4,2)
bar(results(:,2))
set(gca, 'XTickLabel', methods)
title('Fairness (Jain''s Index)')
ylabel('Index')
subplot(1,4,3)
bar(results(:,3))
set(gca, 'XTickLabel', methods)
title('Energy Efficiency')
ylabel('bits/Joule')
subplot(1,4,4)
bar(results(:,4))
set(gca, 'XTickLabel', methods)
title('Utility')
sgtitle('Performance Metrics Comparison for sigma=0.1')

% Plot peer assignments
figure('Position', [500 500 800 600])
for m = 1:length(methods)
    subplot(2,2,m)
    plot3(pos(:,1), pos(:,2), pos(:,3), 'o', 'MarkerFaceColor', 'b');
    hold on
    grid on
    x = positions{m};
    [~, a] = sort(x);
    link_qualities = [];
    for i = 1:N
        j = a(i);
        if j >=1 && j <=N && dist(i,j) <= max_range && j ~= i && energies(i) >= th_energy && energies(j) >= th_energy
            line([pos(i,1) pos(j,1)], [pos(i,2) pos(j,2)], [pos(i,3) pos(j,3)], 'Color', 'r');
            lq = 1 / dist(i,j)^alpha;
            link_qualities = [link_qualities lq];
        end
    end
    title([methods{m} ' Peer Assignments'])
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    avg_lq = mean(link_qualities);
    text(0.05, 0.05, sprintf('Avg LQ: %.4f', avg_lq), 'Units', 'normalized')
end
sgtitle('Peer to Peer Assignments and Link Quality')

% Vary sigma for vs link quality plots
sigmas = 0:0.05:0.5;
num_sigmas = length(sigmas);
results_vs_sigma = zeros(num_sigmas, 4, 4); % sigma x method x metric (throughput, fairness, ee, utility)

for s = 1:num_sigmas
    sigma = sigmas(s);
    fobj = @(x) p2p_cost(x, dist, energies, max_range, th_energy, P, noise, sigma, B, alpha, M);
    
    % Run SAO and PSO for each sigma
    [SAO_Fit, SAO_Pos, ~] = SAO(nAgents, Max_Iter, lb, ub, dim, fobj);
    [PSO_Fit, PSO_Pos, ~] = PSO(nAgents, Max_Iter, lb, ub, dim, fobj);
    NNS_Fit = fobj(NNS_Pos);
    Random_Fit = fobj(Random_Pos);
    
    fits = [SAO_Fit, PSO_Fit, NNS_Fit, Random_Fit];
    pos_list = {SAO_Pos, PSO_Pos, NNS_Pos, Random_Pos};
    
    for m = 1:length(methods)
        x = pos_list{m};
        [~, a] = sort(x);
        rates = zeros(N,1);
        dists_link = zeros(N,1);
        for i = 1:N
            j = a(i);
            if j >=1 && j <=N && dist(i,j) <= max_range && j ~= i && energies(i) >= th_energy && energies(j) >= th_energy
                d = dist(i,j);
                dists_link(i) = d;
                lq = 1 / d^alpha;
                thr = 0;
                for k = 1:M
                    lq_noisy = lq * (1 + sigma * randn());
                    if lq_noisy < 0
                        lq_noisy = 0;
                    end
                    snr = P * lq_noisy / noise;
                    thr = thr + B * log2(1 + snr);
                end
                rates(i) = thr / M;
            end
        end
        throughput = sum(rates);
        non_zero_rates = rates(rates > 0);
        if isempty(non_zero_rates)
            fairness = 1;
        else
            fairness = (sum(non_zero_rates)^2) / (length(non_zero_rates) * sum(non_zero_rates.^2));
        end
        valid_count = length(non_zero_rates);
        if valid_count > 0
            energy_cons = sum(dists_link(rates > 0).^2) + valid_count * P;
        else
            energy_cons = 1;
        end
        ee = throughput / energy_cons;
        utility = -fits(m);
        results_vs_sigma(s, m, :) = [throughput, fairness, ee, utility];
    end
end

% Plot vs link quality (using sigma as proxy for link uncertainty, lower sigma = better quality)
figure('Position', [500 500 800 600])
subplot(1,3,1)
plot(sigmas, squeeze(results_vs_sigma(:,:,3)), 'LineWidth', 2)
legend(methods)
title('Energy Efficiency vs Link Uncertainty')
xlabel('Sigma (Uncertainty)')
ylabel('EE (bits/Joule)')
grid on

subplot(1,3,2)
plot(sigmas, squeeze(results_vs_sigma(:,:,1)), 'LineWidth', 2)
legend(methods)
title('Throughput vs Link Uncertainty')
xlabel('Sigma (Uncertainty)')
ylabel('Throughput (bps/Hz)')
grid on

subplot(1,3,3)
plot(sigmas, squeeze(results_vs_sigma(:,:,2)), 'LineWidth', 2)
legend(methods)
title('Fairness vs Link Uncertainty')
xlabel('Sigma (Uncertainty)')
ylabel('Fairness Index')
grid on
sgtitle('Metrics vs Link Quality Uncertainty')