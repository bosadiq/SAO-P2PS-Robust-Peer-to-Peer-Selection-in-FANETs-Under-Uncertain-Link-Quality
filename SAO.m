function [Agent_Fit, Agent_Pos, Converge_curve] = SAO(Molecules, Max_iter, lb, ub, dim, fobj)

Agent_Pos = zeros(1, dim);
Agent_Fit = inf;

olf = 0.9;
K = 0.6;
T = 5; % Increased for larger steps
M = 0.9;
Step = 0.1; % Adjusted

% Create the initial position of smell molecules
moles_Pos = initialization(Molecules, dim, ub, lb);
Converge_curve = zeros(1, Max_iter);

iter = 0;

% Main loop
while iter < Max_iter
    fitness = zeros(Molecules, 1);
    for k = 1:Molecules        
        % Make Sure smell molecules remains in the search space.
        Clip_ub = moles_Pos(k,:) > ub;
        Clip_lb = moles_Pos(k,:) < lb;
        moles_Pos(k,:) = (moles_Pos(k,:).*(~(Clip_ub + Clip_lb))) + ub.*Clip_ub + lb.*Clip_lb;                      
        % Calculate objective function for each molecules
        fitness(k) = fobj(moles_Pos(k,:));        
        % Agent Fitness
        if fitness(k) < Agent_Fit 
            Agent_Fit = fitness(k); % Update Agent fitness
            Agent_Pos = moles_Pos(k,:);
        end
    end  
    [~, Indes] = max(fitness);
    Worst_Pos = moles_Pos(Indes, :);
    
    % Update the Position of molecules
    for i = 1:Molecules
        current_f = fitness(i);
        r3 = rand();
        r4 = rand();
        r5 = rand();
        
        Sniff_mole = zeros(1, dim);
        for j = 1:dim     
            r1 = rand(); % r1 is a random number in [0,1]
            Sniff_mole(j) = moles_Pos(i,j) + r1 * sqrt(3 * K * T / M); % Sniffing Mode
        end
        fitness_sniff = fobj(Sniff_mole);
        if fitness_sniff < current_f
            moles_Pos(i, :) = Sniff_mole;
            current_f = fitness_sniff;
        end
        if fitness_sniff < Agent_Fit
            Agent_Fit = fitness_sniff;
            Agent_Pos = Sniff_mole;
        end
        
        % Trailing Mode       
        Trail_mole = zeros(1, dim);
        for j = 1:dim     
            Trail_mole(j) = moles_Pos(i,j) + r3 * olf * (moles_Pos(i,j) - Agent_Pos(j)) ...
                - r4 * olf * (moles_Pos(i,j) - Worst_Pos(j)); % Trailing Mode
        end
        fitness_trail = fobj(Trail_mole);
        if fitness_trail < current_f
            moles_Pos(i, :) = Trail_mole;
            current_f = fitness_trail;
        end
        if fitness_trail < Agent_Fit
            Agent_Fit = fitness_trail;
            Agent_Pos = Trail_mole;
        end
        
        % Random Mode
        Random_mole = zeros(1, dim);
        for j = 1:dim     
            Random_mole(j) = moles_Pos(i,j) + r5 * Step; 
        end
        fitness_random = fobj(Random_mole);
        if fitness_random < current_f
            moles_Pos(i, :) = Random_mole;
            current_f = fitness_random;
        end
        if fitness_random < Agent_Fit
            Agent_Fit = fitness_random;
            Agent_Pos = Random_mole;
        end                  
    end
    iter = iter + 1;    
    Converge_curve(iter) = Agent_Fit;
end
end