% This code computes the overlap in post-reward delay distributions under
% different relative pacemaker periods. See Methods section.
% Written 26Sept20 by JGM.

% free parameters
alpha0 = .15;                           % Weber fraction
eps = 0;                                % signal-independent noise
m = [3 6];                              % signal means

ti = m(1):.01:m(2);                     % segment of time domain
t = 0:.01:2*m(2);                       % full time domain

figure(1001)
etaL = [1 4];
for k = 1:2
    eta = etaL(k);                      % POST relative pacemaker period
    alpha = alpha0*eta;                 % Weber fraction for POST
    
    % likelihoods
    s = eps+alpha*m;                    % standard deviations
    like = normpdf(t,m',s');            % distributions
    
    % prior
    m0 = mean(m);                       % mean
    s0 = std(m);                        % standard deviation
    p0 = normpdf(t,m0,s0);              % distribution
    
    % posteriors
    sh = (1./s.^2+1/s0^2).^(-.5);       % standard deviations
    mh = m0+(s0^2./(s.^2+s0^2)).*(m-m0);% means
    p = normpdf(t,mh',sh');             % distributions
    pi = normpdf(ti,mh',sh');           % distributions, segment
    [~,c] = min(abs(diff(pi)));         % intersection of posteriors
    
    % overlapping area
    a(k) = 1-normcdf(ti(c),mh(1),sh(1))+normcdf(ti(c),mh(2),sh(2));
    a(k) = round(a(k),2);
    
    subplot(2,1,k)
    plot(t,p)
    hold on
    plot(t,like,'Color',.7+[0 0 0],'LineWidth',2)
    hold on
    plot(t,p0,'k','LineWidth',2)
    hold on
    plot(ti(c),pi(1,c),'ro')
    hold on
    plot([m; m],[0 10],'k--','LineWidth',1)
    ylim([0 1.1*max(p(:))])
    xlabel('Time (s)')
    ylabel('Probability')
    legend('Small','Large','box','off')
    title(['eta = ' num2str(eta) ', overlap = ' num2str(a(k))])
end