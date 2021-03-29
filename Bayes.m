% This code illustrates the effect of encoding precision on the posterior
% distribution for the case of two options.
% Updated 1Aug20 by JGM.

%-------------------------------------------------------------------------%

figure(1)

s = 4; l = 8;                   % small and large reward
r = 0:.1:12;                    % all possible reward values
sigL = [.8 2];                  % low and high SD (high and low precisions)

p = nan(2,length(r));
maxY = 0;
for e = 1:2
    sig = sigL(e);
    
    % set likelihoods p(s|t), p(l|t)
    ps = normpdf(r,s,sig); ps = ps./sum(ps);
    pl = normpdf(r,l,sig); pl = pl./sum(pl);
    
    % compute prior p(t)
    q = mean([s l]);
    x = std([s l]);
    pr = normpdf(r,q,x); pr = pr./sum(pr);
    
    % compute posteriors
    pos = ps.*pr; pos = pos./sum(pos);
    pol = pl.*pr; pol = pol./sum(pol);
    
    % find posterior means
    [~,v1] = max(pos); [~,v2] = max(pol);
    mps = r(v1); mpl = r(v2);
    
    %-----------------------------% Figure %------------------------------%
    
    subplot(1,2,e)
    
    bl = -.002;                     % for visualization of black segment
    C = [0; .5; .8]*[1 1 1];        % color scheme
    titles = {'High Precision','Low Precision'};
    plot(r,pr,'-.','Color',C(3,:),'LineWidth',2)
    hold on
    plot(r,ps,'--','Color', C(2,:),'LineWidth',2)
    hold on
    plot(r,pl,'--','Color', C(1,:),'LineWidth',2)
    hold on
    plot(r,pos,'-','Color',C(2,:),'LineWidth',5)
    hold on
    plot(r,pol,'-','Color',C(1,:),'LineWidth',5)
    hold on
    plot([mps mpl], bl+[0 0],'k') 	% highlight difference in rewards
   
    xlim([0 max(r)])
    xlabel('Variable')
    ylabel('Probability')
    title(titles{e})
    maxY1 = max([pos pol]);
    maxY = max([maxY1 maxY]);
    ylim([bl 1.03*maxY])
    yticks(0:.01:.05)
    box off
end
legend('Prior','Likelihood, Small','Likelihood, Large',...
        'Posterior, Small','Posterior, Large','Box','Off')