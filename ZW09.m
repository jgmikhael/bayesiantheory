% This code simulates the effect of dopamine on behavior in the rat
% gambling task (Zeeb et al., 2009). 
% Written 3Mar21 by JGM.

%-------------------------------------------------------------------------%

A1 = 2;                         % choice latencies (approximate)
A2 = 10;                        % ITI + collect latencies (approximate)

% free parameters
eta = 10;
beta = 35;                    	% inverse temperature parameter

emp = [4.5 63 10 22.5;          % [saline;
    16.12 41.95 15.13 28.13];   % high DA]
d = [1; 10];                    % DA levels

% likelihood means
r = [1 2 3 4];                  % reward magnitude
to = [5 10 30 40]/eta;         	% time-out duration
p = [.9 .8 .5 .4];              % reward probability
t = A1+(A2+to.*(1-p))./p;      	% time

% likelihood standard deviations
rs = [.001 .001 1 1]; 
ts = [2 2 11 11];

% signal precisions
rli = 1./rs.^2;
tli = 1./ts.^2;

% likelihood precisions
rl = d*rli;
tl = d*tli;

% prior means and precisions
r0 = [mean(r(1:2))*[1 1] mean(r(3:4))*[1 1]];
t0 = [mean(t(1:2))*[1 1] mean(t(3:4))*[1 1]];
rl0 = 1./[var(r(1:2))*[1 1] var(r(3:4))*[1 1]];
tl0 = 1./[var(t(1:2))*[1 1] var(t(3:4))*[1 1]];

% top layer in hierarchy  
r00 = mean(r);
t00 = mean(t);
rl00 = 1./var(r);
tl00 = 1./var(t);

% top layer central tendency (i.e., for the two pairs toward each other)
r0x = r0;
t0x = t0;
r0 = (rl0.*r0+rl00.*r00)./(rl0+rl00);
t0 = (tl0.*t0+tl00.*t00)./(tl0+tl00);
rl0 = rl0+rl00;
tl0 = tl0+tl00;
r = r-r0x+r0;
t = t-t0x+t0;

% posterior means and precisions
rlh = rl+rl0;
tlh = tl+tl0;
rh = (rl.*r+rl0.*r0)./rlh;
th = (tl.*t+tl0.*t0)./tlh;

% reward rate
R = rh./th

% decision rule
px = exp(beta*R)./sum(exp(beta*R),2);

% reward domain
RR = -1:.001:3*max(R(:));

%-------------------------------% Figure %--------------------------------%

figure(101)
C = get(gca,'ColorOrder');      % color scheme
dsh = {'-.','-'};

for q = 1:2                     % saline, DA
    
        rh1 = rh(q,:); rlh1 = rlh(q,:);
        th1 = th(q,:); tlh1 = tlh(q,:);
        
        subplot(2,2,1)
        rr = 0:.01:1.5*max(r);    
        x = normpdf(rr,rh1',1./sqrt(rlh1'));
        for e = 1:4
            hold on
            h(e) = plot(rr,x(e,:),dsh{q},'Color',C(e,:));
            plot(rh1(e)*[1 1],[0 100],dsh{q},'Color',C(e,:),'LineWidth',2)
        end
        xlabel('Reward')
        ylabel('p(reward)')
        box off
        ylim([0 max(x(:))])
        legend(h,'P1','P2','P3','P4','Box','Off')
        
        subplot(2,2,3) 
        tt = 0:.01:1.2*max(t);
        x = normpdf(tt,th1',1./sqrt(tlh1'));
        for e = 1:4
            hold on
          	plot(tt,x(e,:),dsh{q},'Color',C(e,:))
            plot(th1(e)*[1 1],[0 100],dsh{q},'Color',C(e,:),'LineWidth',2)
        end
        xlabel('Time')
        ylabel('p(time)')
        box off
        ylim([0 max(x(:))])

end

subplot(2,2,[2 4])
b = bar(100*px',1,'FaceColor','flat');
b(1).CData = [1 1 1];
b(2).CData = [0 0 0];
hold on
% plot(1:4,emp(1,:),'r')
legend('Baseline DA','High DA','box','off')
box off
xticks(1:4)
xticklabels({'P1','P2','P3','P4'})
y = 0:20:100;
yticks(y)
yticklabels(y)
xlim([1 4]+.5*[-1 1])
ylim([0 100])
ylabel('Percent Choice','FontSize',28)