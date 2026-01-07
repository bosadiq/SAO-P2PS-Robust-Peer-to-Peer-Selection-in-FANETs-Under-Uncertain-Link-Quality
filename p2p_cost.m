% ----------------- p2p_cost.m -----------------
function cost = p2p_cost(x, dist, energies, max_range, th_energy, P, noise, sigma, B, alpha, M)
N = length(x);
a = sort(x);
[~, a] = sort(x);
self_count = 0;
invalid_count = 0;
rates = zeros(N,1);
dists_link = zeros(N,1);
for i = 1:N
    j = a(i);
    if j == i
        self_count = self_count + 1;
    end
    if j <1 || j > N || j == i || dist(i,j) > max_range || energies(i) < th_energy || energies(j) < th_energy
        invalid_count = invalid_count + 1;
        rates(i) = 0;
        dists_link(i) = 0;
    else
        d = dist(i,j);
        dists_link(i) = d;
        lq = 1 / d^alpha;
        thr = 0;
        for m = 1:M
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
utility = 0.5 * throughput + 0.3 * fairness + 0.2 * ee;
cost = -utility + 100 * invalid_count + 100 * self_count;
end
