% This code simulates the effects of temporal and reward encoding
% precisions on behavior in intertemporal choice tasks.
% Written 18Aug20 by JGM.

%-------------------------------------------------------------------------%

% parameters
r = [1 4];                                  % small, large reward magnitude
t = [2 6];                                  % sooner, later trial length
beta = 10;                                  % inverse temperature

rlL = .001:.001:2;                          % likelihood precisions for r
tlL = .001:.001:2;                          % likelihood precisions for t

% priors
r0 = mean(r);                               % reward mean
rl0 = 1./(1+std(r))^2;                      % reward precision
t0 = mean(t);                               % time mean
tl0 = 1./(1+std(t)).^2;                     % time precision

p = nan(length(rlL),length(tlL));           % initialize p
for m = 1:length(rlL)
    for n = 1:length(tlL)
        
        % reward
        rl = rlL(m);                    	% likelihood precision
        rlh = rl+rl0;                       % posterior precision
        rh = (rl.*r+rl0.*r0)./rlh;          % posterior mean
        
        % time
        tl = tlL(n);                       	% likelihood precision
        tlh = tl+tl0;                     	% posterior precision
        th = (tl.*t+tl0.*t0)./tlh;         	% posterior mean
        
        % reward rates
        RS = rh(1)./th(1);                  % small
        RL = rh(2)./th(2);                  % large
        
        % p(selecting large reward)
        p(m,n) = 1./(1+exp(-beta*(RL-RS)));	% softmax
        
    end
end

%-------------------------------% Figure %--------------------------------%

figure(1)

tx = tlL/tl0;
rx = rlL/rl0;

h = pcolor(tx,rx,p);
h.EdgeColor = 'none';
caxis([0 1]);
colormap('gray')
colorbar
xlabel('\lambda_t/\lambda_{t0}','Interpreter', 'tex')
ylabel('\lambda_r/\lambda_{r0}','Interpreter', 'tex')
xlim([min(tlL) 10])
ylim([min(rlL) 10])
xticks(0:5:5*round(max(tx)/5))
yticks(0:5:5*round(max(rx)/5))