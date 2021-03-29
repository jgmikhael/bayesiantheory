% This code simulates the correlation between timing precision and measures
% of impulsive choice (Marshall et al., 2014).
% Written 17Aug20 by JGM.

%-------------------------------------------------------------------------%

% parameters
de = [.5 1.5];           	% min and max DA levels, fluctuating around 1
sr = .8;                    % signal-independent noise for rewards
st = 2.7;                	% signal-independent noise for time
beta = 180;              	% inverse temperature for choice rule
eta = 10;                   % POST relative pacemaker period

% make DA levels logarithmically spaced for illustration
d = logspace(log10(de(1)),log10(de(2)),15);

%-------------------------------------------------------------------------%

% (1) intertemporal choice task

r = [1 3];                              % reward magnitudes
iti = 120;                              % intertrial interval
t = [2.5 5 10 30; 30 30 30 30];         % duration for [short; long]

% reward prior
r0 = mean(r);                           % mean
rl0 = 1./(1+var(r));                    % precision

% time prior
t0 = mean(t);                           % mean
tl0 = 1./(1+var(t));                    % precision

% initialize
p = nan(length(d),length(t));           % probability of choosing Large
th = nan(2,length(t));                  % estimated time ('h' means 'hat')

figure(101)
rr = 0:.1:1.5*r(2);                     % range of rewards for illustration
tt = 0:.1:1.5*t(end);                   % range of times for illustration
ttl = {'Low DA','Medium DA','High DA'};
for q = 1:length(d)
   
    % effect of DA on likelihood standard deviations
    rs = sr./d(q);                      % reward
    ts = st./d(q);                      % time
    
    % reward
    rl = 1./rs.^2;                      % likelihood precisions
    rlh = rl+rl0;                     	% posterior precisions
    rh = (rl.*r+rl0.*r0)./rlh;        	% posterior means
    
    % interval
    tl = 1./ts.^2;                      % likelihood precisions
    tlh = tl+tl0;                     	% posterior precisions
    th = (tl.*t+tl0.*t0)./tlh;          % posterior means
    
    % reward rates
    RS = rh(1)./(th(1,:)+iti/eta);    	% small
    RL = rh(2)./(th(2,:)+iti/eta);     	% large
    
    % probability of selecting the large reward
    p(q,:) = 1./(1+exp(-beta*(RL-RS)));	% softmax

    % select [lowest middle highest] DA levels for illustration
    g = [1 floor(length(d)/2) length(d)];  
    [gm1, gm] = min(abs(q-g));          % find q for which 1 of the 3 hold
    
    if gm1 == 0 % if current DA level is lowest, middle, or highest
        
        subplot(6,3,gm)                 % visualize reward central tendency
        x = normpdf(rr,rh',1./sqrt(rlh'));
        plot(rr,x)
        hold on
        plot(rh(1)*[1 1],[0 10],'k--','LineWidth',2)
        plot(rh(2)*[1 1],[0 10],'k--','LineWidth',2)
        plot(r(1)*[1 1],[0 10],'k','LineWidth',2)
        plot(r(2)*[1 1],[0 10],'k','LineWidth',2)
        xlabel('Reward (Numerator)')
        ylabel('p(reward)')
        box off
        ylim([0 max(x(:))])
        title(ttl{gm})

        for k = 1:4                     % for each delay pair
            subplot(6,3,gm+3*k)         % visualize time central tendency
            x = normpdf(tt,th(:,k),1./sqrt(tlh(:,k)));
            plot(tt,x)
            hold on
            plot(th(1,k)*[1 1],[0 10],'k--','LineWidth',2)
            plot(th(2,k)*[1 1],[0 10],'k--','LineWidth',2)
            plot(t(1,k)*[1 1],[0 10],'k','LineWidth',2)
            plot(t(2,k)*[1 1],[0 10],'k','LineWidth',2)
            xlabel('Time (Denominator)')
            ylabel('p(time)')
            box off
            ylim([0 max(x(:))])
            legend(['s = ' num2str(t(1,k))],'box','off')
        end
    end
end
subplot(6,3,16:18)
plot(t(1,:),p)
xlabel('Short Delay')
ylabel('p(Long)')
legend('Lowest DA','Location','Northwest','box','off')

LOGLL = mean(log(p./(1-p)),2);          % mean log odds

%-------------------------------------------------------------------------%

% (2) timing (bisection) task

t = [4 12];                             % short and long temporal intervals

% time prior
t0 = mean(t);                           % mean
tl0 = 1./(1+var(t));                    % precision

tt = 0:.1:20;                          	% time domain for illustration

% initialize
pL = nan(length(d),length(tt));         % probability of choosing Long
maxY = 0;                              	% for ylim

C = linspace(0,.8,length(d))'*[1 1 1];	% color scheme

for q = 1:length(d)

	% effect of DA on likelihood standard deviations
  	ts = st./d(q);                      % time

    tl = 1./ts.^2;                      % likelihood precisions
    tlh = tl+tl0;                     	% posterior precisions
    th = (tl.*t+tl0.*t0)./tlh;          % posterior means
    
    a = normpdf(tt,th(2),1/sqrt(tlh));
    b = normpdf(tt,th(1),1/sqrt(tlh));
    pL(q,:) = a./(a+b);
    
    figure(102)
    subplot(4,1,1)
    plot(tt,a,'Color',C(q,:))
    hold on
    plot(tt,b,'Color',C(q,:))
    plot(t(1)*[1 1],[0 100],'k--')
    plot(t(2)*[1 1],[0 100],'k--')
    maxY = max(max(b),maxY);
end
xlim([0 tt(end)])
ylim([0 maxY])
xlabel('Time')

subplot(4,1,2)
for q = 1:length(d)
    plot(tt,pL(q,:),'Color',C(q,:))
    hold on
end
plot([min(tt) max(tt)],[.5 .5],'k--')
xlabel('Time')
ylabel('p(Long)')
xticks([t(1) mean(t) t(2)])
yticks([0 .5 1])

% examine p(LL) between min(t) and max(t) to model as logit
tt1 = min(t):.1:max(t);
[~,ind1] = min(abs(tt-min(t)));
[~,ind2] = min(abs(tt-max(t)));
pL1 = pL(:,ind1:ind2);

% infer temperature parameter
y = log(pL1./(1-pL1));
bb = nan(1,size(y,1));
sigma = nan(1,size(y,1));
for e = 1:size(y,1)
    prms = polyfit(y(e,:),tt1,1);       % if y = 1/(1+exp(-x/sigma))...
    sigma(e) = prms(1);                 % ... then x = sigma*ln(y/(1-y))
    bb(e) = prms(2);
end

subplot(4,1,3)
for q = 1:size(y,1)
    plot(tt1,y(q,:),'Color',C(q,:))
    hold on
    plot(tt1,(tt1-bb(q))./sigma(q),'--','Color',C(q,:))
end
xlabel('Time')
ylabel('log(p/(1-p))')

subplot(4,1,4)
plot(d,sigma,'k')
hold on
xlabel('DA')
ylabel('sigma')

%-------------------------------% Figure %--------------------------------%

figure(103)
subplot(1,2,1)
plot(d,LOGLL)
title('LOGLL')
subplot(1,2,2)
plot(d,sigma)
title('sigma')
for e = 1:2
    subplot(1,2,e)
    xlabel('DA level')
    xlim([min(d) max(d)])
end

figure(1)
scatter(LOGLL,sigma,400,'k','filled')
xticks([-1.5 0 1.5 3])
yticks([.5 1.5 2.5 3.5])
set(gca,'FontSize',28);
xlabel('MEAN LOG ODDS (LL)','FontSize',30)
ylabel('BISECTION \sigma','Interpreter', 'tex','FontSize',30)
box off

if 1
    xlim([-1.5 3])
    ylim([.4 3.65])  
else % put box around region where data empirically observed 
    hold on
    plot([-1.5 3],[.5 .5],'k')
    plot([-1.5 3],[3.5 3.5],'k')
    plot([-1.5 -1.5],[.5 3.5],'k')
    plot([3 3],[.5 3.5],'k')
    plot([0 0],[.5 3.5],'k')
end