% This code simulates the effect of block order history.
% Written 1Mar21 by JGM.

%-------------------------------------------------------------------------%

% parameters
dag = [0 1 2 5];            % DA agonist in arbitrary units
d = 1+dag;                  % total DA level
rs = .5;                    % encoding SD for rewards
ts = 2;                     % encoding SD for time
g = 10;                     % gain on temporal SD for low precision
eta = 10;                   % POST relative pacemaker period
gamma = .99;              	% decay rate for influence of recent history
beta = 20;                  % inverse temperatures
a = 1;                      % strength of learning prior (stickiness)
rLarge = 3;                 % number of pellets for larger option
trial = 50;                 % total trial length
maxPRE = 50;                % max PRE

%-------------------------------------------------------------------------%

C = [0 .3 .6 .8]'*[1 1 1];              % color scheme
ylabels = 0:10:maxPRE;
n = 101;                                % number of timepoints
PRExx = linspace(0,maxPRE,n);           % timepoints for larger option

%-------------------------------------------------------------------------%

% rewards, for [small large]
r = [1 rLarge];                         % encoding means
rli = 1./rs.^2;                         % encoding precision
r0 = mean(r);                           % prior mean
rl0 = 1./var(r);                        % prior precision
rr = 0:.01:1.5*max(r);                  % reward domain for illustration

figure(101)
for cond = 1:2                          % for [Asc  Desc]
    
    % encoding precision
    PREli = 1./(ts.*g).^2;
    
    % learning priors
    PREx0 = [];
    for ee = 1:n
        if cond == 1                    % no recent-history bias
            w = zeros(1,n); w(ee) = 1;
        else                            % recent-history bias
            w = gamma.^(1:n);
            w(1:ee-1) = 0; 
        end
        PREx0(ee) = w*PRExx'/sum(w);    % learning prior means
    end
    PRExl0 = a*PREli;                   % learning prior precisions
    
    % likelihood precision and means
    PRElii = PREli + PRExl0;
    PREx = (PREli.*PRExx+PRExl0.*PREx0)./PRElii;
    
    figure(102)
    subplot(1,2,1)
    plot(gamma.^(1:n))
    xlabel('Blocks')
    ylabel('Weight')
    title('Weights of Previous Blocks')
    box off
    subplot(1,2,2)
    hold on
    plot(PRExx,'g--')
    plot(PREx0,'r--')
    plot(PREx,'b--')
    xlabel('PRE')
    ylabel('Learned Estimates')
    
    PRE = [0*PREx; PREx];               % likelihood means
    PRE0 = mean(PRE);                   % prior mean
    PREl0 = 1./(1+var(PRE));          	% prior precision
    
    % POST
    POST = (trial-PRE)/eta;            	% likelihood mean
    POSTli = 1./(ts.*g).^2;             % likelihood precision 

    POST0 = mean(POST);               	% prior mean
    POSTl0 = 1./(1+var(POST));         	% prior precision
    tt = -10:.01:(max(PREx)+max(POST));	% time domain for illustration
    
    p = zeros(length(d),n);             % initialize p(LL)
    for q = 1:length(d)
        
        % effect of DA on likelihood precisions
        rl = rli.*d(q).^2;
        PREl = PRElii.*d(q).^2;
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
        
        figure(102)
        plot(PREh(2,:),'Color',C(q,:))
        legend('True PRE','Sticky Prior','Encoded PRE',...
        'Location','Northwest','box','off')
        
        % reward rates
        RS = rh(1)./(PREh(1,:)+POSTh(1,:));       	% small
        RL = rh(2)./(PREh(2,:)+POSTh(2,:));        	% large
        
        % p(selecting large reward)
        p(q,:) = 1./(1+exp(-beta*(RL-RS)));         % softmax
        
        if q == 1 || q == length(d)                 % lowest, highest DA
            figure(101)
            subplot(4,2,1+4*(cond-1))
            x = normpdf(rr,rh',1./sqrt(rlh'));
            plot(rr,x,'Color',C(q,:))
            hold on
            plot(rh(1)*[1 1],[0 10],'Color',C(q,:),'LineWidth',2)
            hold on
            plot(rh(2)*[1 1],[0 10],'Color',C(q,:),'LineWidth',2)
            hold on
            xlabel('Reward (Numerator)')
            ylabel('p(reward)')
            box off
            ylim([0 max(x(:))])
            
            subplot(4,2,3+4*(cond-1))   % illustrate for middle time
            mid = ceil(length(PREh)/2);
            tmid = PREh(:,mid)+POSTh(:,mid);
            SDmid = sqrt(1./PRElh(:,mid)+1./POSTlh(:,mid));
            x = normpdf(tt,tmid,SDmid);
            plot(tt,x,'Color',C(q,:))
            hold on
            plot(tmid(1)*[1 1],[0 10],'Color',C(q,:),'LineWidth',2)
            hold on
            plot(tmid(2)*[1 1],[0 10],'Color',C(q,:),'LineWidth',2)
            hold on
            xlabel('Time (Denominator)')
            ylabel('p(time)')
            box off
            xlim([.5*tmid(1) 1.5*tmid(2)])
            ylim([0 max(x(:))])
        end
    end
    baseline(cond,:) = p(1,:);       	% baseline (saline) levels

        subplot(4,2,4*cond+[-2 0])
        
        for e = 1:length(d)             % for legend
            DAlevels{e} = [num2str(dag(e)) ' a.u.'];
        end
        
        for e = 1:size(p,1)
            h(e) = plot(100*p(e,:)','Color',C(e,:));
            hold on
        end
        legend(h,DAlevels,'Box','Off','FontSize',25)
        xlabel('delay (sec)')
        ylabel('% choice of the large reinforcer')
      	yticks(0:20:100)
        xticks(floor(ylabels*n/maxPRE))
        xlim([1-.2 n+.2])
        xticklabels(ylabels)
        ylim([0 100])
        box off
    
end