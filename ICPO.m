function [Best_score, Best_pos, FES_curve] = ICPO(SearchAgents, lb, ub, dim, fobj, maxFES)

fprintf('ICPO (Pure FES control): dim=%d, SearchAgents=%d, maxFES=%d\n', ...
    dim, SearchAgents, maxFES);

% 初始化搜索边界
lb = ones(1, dim) .* lb;
ub = ones(1, dim) .* ub;

% 初始化控制参数（完全保持原样）
N_min = 80;     % 种群规模最小值
T = 2;          % 循环数量
Tf = 0.5;       % 第三和第四防御机制之间的权衡百分比

% 改进1：使用拉丁超立方初始化
X = lhs_init(SearchAgents, dim, lb, ub);
fitness = zeros(1, SearchAgents);

% 计算初始适应度（消耗 SearchAgents 次 FES）
FES = 0;
for i = 1:SearchAgents
    fitness(i) = fobj(X(i,:));
    FES = FES + 1;
end

% 初始化全局最优
[Best_score, index] = min(fitness);
Best_pos = X(index, :);

% 存储每个峰冠豪猪个人最佳位置
Xp = X;

% 用于记录 FES-Best 曲线（预分配足够空间）
max_records = 10000;
FES_curve = zeros(max_records, 2);  % 第1列: FES, 第2列: Best_score
record_idx = 1;
FES_curve(record_idx,:) = [FES, Best_score];
record_idx = record_idx + 1;

% Golden-Sine 全局项（在循环外预先生成参数，循环内只更新 gold_factor）
t_golden = (sqrt(5)-1)/2; % 黄金比例
a = -pi; b = pi;
x1 = a*(1 - t_golden) + b*t_golden;
x2 = a*t_golden + b*(1 - t_golden);

while FES < maxFES
    
    % 计算本轮剩余可用 FES
    remaining_FES = maxFES - FES;
    if remaining_FES <= 0
        break;
    end
    this_iter_agents = min(SearchAgents, remaining_FES);
    
    % 本轮 Golden-Sine 全局引导项（每完整一轮更新一次）
    r1_g = 2*pi*rand;
    r2_g = pi*rand;
    gold_factor = r2_g * sin(r1_g);
    r2 = rand;
    
    for i = 1:this_iter_agents
        U1 = rand(1, dim) > rand;
        beta_levy = 1.3 + 0.5*min(1, FES/maxFES);  % 改进2：动态 Levy β
        if rand < rand  % 探索阶段
            if rand < rand  % 第一防御机制
                rand_index = randi(SearchAgents);
                y = (X(i,:) + X(rand_index,:)) / 2;
                
                % Golden-Sine 改进项（原代码逻辑完全保留）
                gold_term = gold_factor .* abs(x1*Best_pos - x2*X(i,:));
                if rand < 0.5
                    X(i,:) = X(i,:) + randn .* abs(2*rand*Best_pos - y) + gold_term;
                else
                    X(i,:) = X(i,:) + randn .* abs(2*rand*Best_pos - y) - gold_term;
                end
            else  % 第二防御机制
                
                rand_index1 = randi(SearchAgents);
                rand_index2 = randi(SearchAgents);
                y = (X(i,:) + X(rand_index1,:)) / 2;
                X(i,:) = U1 .* X(i,:) + (1-U1) .* (y + LevyFlight(beta_levy, dim).*(X(rand_index1,:) - X(rand_index2,:)));
            end
        else  % 开发阶段
            Yt = 2*rand*(1 - min(1, FES/maxFES))^(min(1, FES/maxFES));
            U2 = (rand(1,dim) < 0.5)*2-1;
            
            if rand < Tf  % 第三防御机制
                
                S0 = LevyFlight(beta_levy, dim).*U2;
                St = exp(fitness(i)/(sum(fitness)+eps));
                S1 = S0.*Yt.*St;
                rand_index1 = randi(SearchAgents);
                rand_index2 = randi(SearchAgents);
                rand_index3 = randi(SearchAgents);
                X(i,:) = (1-U1).*X(i,:) + U1.*(X(rand_index1,:) + St.*(X(rand_index2,:) - X(rand_index3,:)) - S1);
            else  % 第四防御机制
                Mt = exp(fitness(i)/(sum(fitness)+eps));
                vt = X(i,:);
                rand_index = randi(SearchAgents);
                Vtp = X(rand_index,:);
                Ft = rand(1,dim).*(Mt.*(-vt+Vtp));
                S2 = rand*U2.*Yt.*Ft;
                
                % 改进3：自适应 alpha（原逻辑完全保留，使用前一轮 Best 进行判断）
                if record_idx > 2  % 至少有前一记录
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
        
        % 边界检查
        X(i,:) = max(X(i,:), lb);
        X(i,:) = min(X(i,:), ub);
        
        % 计算新适应度（消耗一次 FES）
        new_fitness = fobj(X(i,:));
        FES = FES + 1;
        
        % 更新个体最优和全局最优
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
    
    % 本轮更新完成后记录当前 FES 和 Best_score
    if record_idx <= max_records
        FES_curve(record_idx,:) = [FES, Best_score];
        record_idx = record_idx + 1;
    end
    
    % 进度输出
    if mod(FES, 1000) < this_iter_agents || FES >= maxFES
        fprintf('FES: %6d/%d (%.1f%%), Best: %.4e\n', ...
            FES, maxFES, FES/maxFES*100, Best_score);
    end
    
    % 动态调整种群规模（原逻辑完全不变）
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

% 裁剪曲线
FES_curve = FES_curve(1:record_idx-1, :);

end