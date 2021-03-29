% This code simulates the effect of dopamine on behavior in intertemporal
% choice tasks (Cardinal et al., 2000; Tanno et al., 2014) and probability
% discounting tasks (St Onge et al., 2010). It also reproduces the saline
% conditions in Fig. 3B,C of Cardinal et al. (2000) and Figs. 3A and 4A of
% St Onge et al. (2010), each within a single figure.
% Written 1Mar21 by JGM.

%-------------------------------------------------------------------------%

% parameters for [ITC PD]
rs = [.5 2];                	% encoding SD for rewards
ts = [4 7];                     % encoding SD for time
eta = 10;                       % POST relative pacemaker period
beta = [30 10];                 % inverse temperatures
dag = {[0 .5 2 5],[0 10]};      % DA agonists (a.u.)
g = [1 5; 1 10];                % gain on temporal SD; [high low] precision
rLarge = [4 10];                % number of pellets for larger option

%-------------------------------------------------------------------------%

n = 101;                        % number of sampled points
pre2i = 7;                      % pre for a single trial in PD
px = [linspace(1,1,n);
linspace(.125,1,n)];            % reward probabilities in [ITC; PD]

% properties of the task for [ITC PD]
trial = [100 40];               % total trial length
maxPRE = [60 pre2i]./min(px'); 	% max PRE
s = [0 pre2i];                  % PRE for small option

PREx = [
    linspace(0,60,n);               % ITC
    pre2i+trial(2).*(1./px(2,:)-1)	% PD      
    ];

%-------------------------------------------------------------------------%

% for figures
xlbls = {'Delay to large reinforcer (s)',...
    'Large/risky lever probability by block'};
ylbls = {'Percent choice of large reinforcer',...
    '% choice of Large/Risky lever'};
lgdLocation = {'Northeast','Southwest'};
xtlbls = {'100','50','25','12.5%';'12.5','25','50','100%'};
tickY = [0 20 100; 0 25 100];
tickX = {0:10:maxPRE(1),[12 25 50 100]};
x = [PREx(1,:); 1:n];	% domain on x-axis
limX = [0 maxPRE(1); 1 n];

% for ITC legend
for e = 1:length(dag{1})
    v1{e} = [num2str(dag{1}(e)) ' a.u.'];
end
DAlevels = {v1,{'Baseline DA','High DA'}};

% color scheme
C0 = {[0 .3 .6 .8]      % ITC
    [0 .5]};            % PD

%-------------------------------------------------------------------------%

for expt = 1:2                      	% for [ITC PD]
    h = []; baseline = [];          	% initialize
    C = C0{expt}'*[1 1 1];              % color scheme
    d = 1+dag{expt};                  	% total DA level
    pxx = [ones(1,n); px(expt,:)];
    
    % rewards, for [small large]
    r = [1 rLarge(expt)];            	% encoding means
    rli = 1./(1+rs(expt).^2);          	% encoding precision
    r0 = mean(r);                     	% prior mean
    rl0 = 1./var(r);                 	% prior precision
    rr = 0:.01:1.5*max(r);          	% reward domain, for illustration

    for cond = 1:2                  	% for [high low] temporal precision
        
        % PRE
        PREli = 1./(ts(expt).*g(expt,cond)).^2;   	% likelihood precision
        PRE = [s(expt)+zeros(1,n); PREx(expt,:)]; 	% likelihood means
        PRE0 = mean(PRE);                          	% prior mean
        PREl0 = 1./(1+var(PRE));                   	% prior precision
        
        % POST
        POST = (trial(expt)./pxx-PRE)/eta;          % likelihood mean
        POSTli = 1./(eta*ts(expt).*g(expt,cond)).^2;% likelihood precision
        
        POST0 = mean(POST);                      	% prior mean
        POSTl0 = 1./(1+var(POST));              	% prior precision
        tt = -10:.01:(max(PREx(expt,:))+max(POST));	% time domain, for illustration
        
        p = zeros(length(d),n);                     % initialize p(LL)
        for q = 1:length(d)
            
            % effect of DA on likelihood precisions
            rl = rli.*d(q).^2;
            PREl = PREli.*d(q).^2;
            POSTl = POSTli.*d(q).^2;
            
            % reward
            rlh = rl+rl0;                            	% posterior precisions
            rh = (rl.*r+rl0.*r0)./rlh;                  % posterior means
            
            % PRE
            PRElh = PREl+PREl0;                      	% posterior precisions
            PREh = (PREl.*PRE+PREl0.*PRE0)./PRElh;    	% posterior means
            
            % POST
            POSTlh = POSTl+POSTl0;                     	% posterior precisions
            POSTh = (POSTl.*POST+POSTl0.*POST0)./POSTlh;% posterior means
            
            % reward rates
            RS = rh(1)./(PREh(1,:)+POSTh(1,:));       	% small
            RL = rh(2)./(PREh(2,:)+POSTh(2,:));        	% large
            
            % p(selecting large reward)
            p(q,:) = 1./(1+exp(-beta(expt)*(RL-RS)));	% softmax
            
            % illustration of posteriors under baseline vs high DA
            if q == 1 || q == length(d)     % lowest, highest DA
                figure(100+expt)
                subplot(4,2,1+4*(cond-1))
                hold on
                y = normpdf(rr,rh',1./sqrt(rlh'));
                plot(rr,y,'Color',C(q,:))
                plot(rh(1)*[1 1],[0 10],'Color',C(q,:),'LineWidth',2)
                plot(rh(2)*[1 1],[0 10],'Color',C(q,:),'LineWidth',2)
                xlabel('Reward (Numerator)')
                ylabel('p(reward)')
                box off
                ylim([0 max(y(:))])
                
                subplot(4,2,3+4*(cond-1))   % illustrate for middle time
                hold on
                mid = ceil(length(PREh)/2);
                tmid = PREh(:,mid)+POSTh(:,mid);
                SDmid = sqrt(1./PRElh(:,mid)+1./POSTlh(:,mid));
                y = normpdf(tt,tmid,SDmid);
                plot(tt,y,'Color',C(q,:))
                plot(tmid(1)*[1 1],[0 10],'Color',C(q,:),'LineWidth',2)
                plot(tmid(2)*[1 1],[0 10],'Color',C(q,:),'LineWidth',2)
                xlabel('Time (Denominator)')
                ylabel('p(time)')
                box off
                xlim([.5*tmid(1) 1.5*tmid(2)])
                ylim([0 max(y(:))])
            end
        end
        baseline(cond,:) = p(1,:);          % baseline (no-agonist) levels
        
        subplot(4,2,4*cond+[-2 0])
        hold on

        % flip the x-axis for the PD descending condition
        if expt == 2 && cond == 1; p = fliplr(p); end 

        % plot the curves
        h = plot(x(expt,:),100*p);

        % figure properties
        set(h, {'color'}, num2cell(C,2));
        legend(DAlevels{expt},'Box','Off',...
            'Location',lgdLocation{expt},'FontSize',25)
        xlim(limX(expt,:))
        ylim([0 100])
        if expt == 2
            if cond == 1;   xticks(fliplr(n-tickX{expt}))
            else;           xticks(tickX{expt}); end
            xticklabels(xtlbls(cond,:));
        end
        yticks(tickY(expt,1):tickY(expt,2):tickY(expt,3))       
        xlabel(xlbls{expt},'FontSize',25)
        ylabel(ylbls{expt},'FontSize',25)
        box off
        
    end
    
    % baseline
    figure(103)
    subplot(1,2,expt)
    
    % plot the curves
    h = plot(x(expt,:),100*baseline);
    
    % figure properties
    set(h, {'color'}, num2cell(C0{2}'*[1 1 1],2));
    if expt == 2; xticklabels(xtlbls(cond,:)); end
    xticks(tickX{expt});
    xlim(limX(expt,:))
    yticks(tickY(expt,1):tickY(expt,2):tickY(expt,3))
    xlabel(xlbls{expt},'FontSize',25)
    ylabel(ylbls{expt},'FontSize',25)
    ylim([0 100])
    legend('High Temporal Precision','Low Temporal Precision',...
        'Location',lgdLocation{expt},'Box','Off','FontSize',25)
    box off
    
end

%-------------------------------------------------------------------------%

% reproduce the saline conditions in Cardinal et al. (2000), Fig. 3

figure(104)
subplot(1,2,1)

salineITC = [82 56.8 40 29 22;  % cue
    76.2 54.8 44.8 39.5 28.4];  % no cue

% plot the curves
h = plot(salineITC','-o');

% figure properties
set(h, {'color'}, num2cell(C0{2}'*[1 1 1],2));
legend('Cue','No Cue','Box','Off','FontSize',25)
xlabel(xlbls{1},'FontSize',25)
ylabel(ylbls{1},'FontSize',25)
xticks(1:5); xticklabels([0 10 20 40 60])
yticks(0:10:100)
xlim([1 5])
ylim([0 100])
box off

%-------------------------------------------------------------------------%

% reproduce the saline conditions in St Onge et al. (2010), Figs. 3A & 4A

figure(104)
subplot(1,2,2)

% empirical values
salinePD = [56 69 82 89;    % descending
    55 78 95 98];           % ascending

% plot the curves
h = plot(salinePD');

% figure properties
hold on
for e = 1:2
  scatter(1:4,salinePD(e,:),450,'o','filled',...
       'MarkerFaceColor',[1 1 1],'MarkerEdgeColor',C0{2}(e)+[0 0 0]);
end
set(h, {'color'}, num2cell(C0{2}'*[1 1 1],2));
xlim([0.6 4.4])
ylim([0 100])
yticks(0:25:100)
xlabel(xlbls{2},'FontSize',25)
ylabel(ylbls{2},'FontSize',25)
xticks(1:4); xticklabels(xtlbls(2,:))
legend(h,'Descending', 'Ascending','Box','Off',...
    'Location','Southwest','FontSize',25)
box off