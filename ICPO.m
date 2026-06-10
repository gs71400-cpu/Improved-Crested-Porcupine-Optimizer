function [Best_score, Best_pos, FES_curve] = ICPO(SearchAgents, lb, ub, dim, fobj, maxFES)

fprintf('ICPO (Pure FES control): dim=%d, SearchAgents=%d, maxFES=%d\n', ...
    dim, SearchAgents, maxFES);


lb = ones(1, dim) .* lb;
ub = ones(1, dim) .* ub;


N_min = 80;     
T = 2;          
Tf = 0.5;       

X = lhs_init(SearchAgents, dim, lb, ub);
fitness = zeros(1, SearchAgents);


FES = 0;
for i = 1:SearchAgents
    fitness(i) = fobj(X(i,:));
    FES = FES + 1;
end


[Best_score, index] = min(fitness);
Best_pos = X(index, :);


Xp = X;


max_records = 10000;
FES_curve = zeros(max_records, 2);  
record_idx = 1;
FES_curve(record_idx,:) = [FES, Best_score];
record_idx = record_idx + 1;


t_golden = (sqrt(5)-1)/2; % 黄金比例
a = -pi; b = pi;
x1 = a*(1 - t_golden) + b*t_golden;
x2 = a*t_golden + b*(1 - t_golden);

while FES < maxFES
    
   
    remaining_FES = maxFES - FES;
    if remaining_FES <= 0
        break;
    end
    this_iter_agents = min(SearchAgents, remaining_FES);
    
   
    r1_g = 2*pi*rand;
    r2_g = pi*rand;
    gold_factor = r2_g * sin(r1_g);
    r2 = rand;
    
    for i = 1:this_iter_agents
        U1 = rand(1, dim) > rand;
        beta_levy = 1.3 + 0.5*min(1, FES/maxFES);  
        if rand < rand  
            if rand < rand  
                rand_index = randi(SearchAgents);
                y = (X(i,:) + X(rand_index,:)) / 2;
                
                
                gold_term = gold_factor .* abs(x1*Best_pos - x2*X(i,:));
                if rand < 0.5
                    X(i,:) = X(i,:) + randn .* abs(2*rand*Best_pos - y) + gold_term;
                else
                    X(i,:) = X(i,:) + randn .* abs(2*rand*Best_pos - y) - gold_term;
                end
            else  
                
                rand_index1 = randi(SearchAgents);
                rand_index2 = randi(SearchAgents);
                y = (X(i,:) + X(rand_index1,:)) / 2;
                X(i,:) = U1 .* X(i,:) + (1-U1) .* (y + LevyFlight(beta_levy, dim).*(X(rand_index1,:) - X(rand_index2,:)));
            end
        else  
            Yt = 2*rand*(1 - min(1, FES/maxFES))^(min(1, FES/maxFES));
            U2 = (rand(1,dim) < 0.5)*2-1;
            
            if rand < Tf  
                
                S0 = LevyFlight(beta_levy, dim).*U2;
                St = exp(fitness(i)/(sum(fitness)+eps));
                S1 = S0.*Yt.*St;
                rand_index1 = randi(SearchAgents);
                rand_index2 = randi(SearchAgents);
                rand_index3 = randi(SearchAgents);
                X(i,:) = (1-U1).*X(i,:) + U1.*(X(rand_index1,:) + St.*(X(rand_index2,:) - X(rand_index3,:)) - S1);
            else  
                Mt = exp(fitness(i)/(sum(fitness)+eps));
                vt = X(i,:);
                rand_index = randi(SearchAgents);
                Vtp = X(rand_index,:);
                Ft = rand(1,dim).*(Mt.*(-vt+Vtp));
                S2 = rand*U2.*Yt.*Ft;
                
                if record_idx > 2  
                    prev_best = FES_curve(record_idx-2, 2);
                    improvement = abs(prev_best - Best_score) / (abs(prev_best) + eps);
                    if improvement > 0.1
                        alpha = 0.2;
                    else
                        alpha = 0.2 * (0.5 + 0.5 * cos(pi * min(1, FES/maxFES)));
                    end
                else
                    alpha = 0.2;
                end
                
                X(i,:) = Best_pos + (alpha*(1-r2)+r2)*(U2.*Best_pos - X(i,:)) - S2;
            end
        end
        
        
        X(i,:) = max(X(i,:), lb);
        X(i,:) = min(X(i,:), ub);
        

        new_fitness = fobj(X(i,:));
        FES = FES + 1;
        
       
        if new_fitness < fitness(i)
            Xp(i,:) = X(i,:);
            fitness(i) = new_fitness;
            if new_fitness < Best_score
                Best_pos = X(i,:);
                Best_score = new_fitness;
            end
        else
            X(i,:) = Xp(i,:);
        end
    end
    
  
    if record_idx <= max_records
        FES_curve(record_idx,:) = [FES, Best_score];
        record_idx = record_idx + 1;
    end
    
  
    if mod(FES, 1000) < this_iter_agents || FES >= maxFES
        fprintf('FES: %6d/%d (%.1f%%), Best: %.4e\n', ...
            FES, maxFES, FES/maxFES*100, Best_score);
    end
    
   
    cycle_length = maxFES / T;
    current_cycle_progress = rem(FES, cycle_length) / cycle_length;
    New_SearchAgents = fix(N_min + (SearchAgents - N_min) * (1 - current_cycle_progress));
    
    if New_SearchAgents < SearchAgents
        [~, sorted_indexes] = sort(fitness);
        X = X(sorted_indexes(1:New_SearchAgents), :);
        Xp = Xp(sorted_indexes(1:New_SearchAgents), :);
        fitness = fitness(sorted_indexes(1:New_SearchAgents));
        SearchAgents = New_SearchAgents;
    end
    
end

FES_curve = FES_curve(1:record_idx-1, :);

end
