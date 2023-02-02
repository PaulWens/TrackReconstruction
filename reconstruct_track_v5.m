function [xmean,stats] = reconstruct_track_v5(t,p,h,s,tb,m,sigmaw,tF,XF,q,sigmaFall,nuFall,type,Xw,tsurf,nburnin,nsamples,thin,initsigma,workingDir,doParallel,plots)
% Track reconstruction from position fixes and dead-reckoning data. The 
% model described in [1] is fitted to the data via MCMC in JAGS. 
%
% Version 5. Model structure with observation model for FGPS positions fixes
% and a simplified observation model for whale positions (lat,lon) obtained 
% from visual tracking. Thus ignores the effect of range on observation error.
%
% Output arguments:
%  xmean: the most-probable track - posterior means of animal positions x(i), in relative Cartesian coordinates
%  stats: data structure with posterior summaries and convergence parameter Rhat for parameters x (animal position), vcor (velocity correction), and sigma (velocity correction process error SDs)
%
% Input arguments:
%  t: time vector for 1 Hz tag data
%  p,h: pitch and heading of animal at times t
%  s: forward speed of animal at times t. Can be vector or, to assume constant speed, a scalar. Using a vector with speed estimates (e.g. based on flow noise) decreases runtime and improves convergence. Default is 1 m/s
%  tb: time of visual position fixes on the animal made from the observation boat
%  m: ranging method index for each visual fix (1: naked eye, 2: laser-range finder)
%  sigmaw: visual observation error SDs in km, for sightings made by eye and laser-range finder. The same value is applied to the x and y dimensions. Default is [0.075 0.025] based on field tests
%  tF: time of position fixes on the animal made with the Fastloc-GPS logger
%  XF: position of animal based on Fastloc GPS
%  q: quality index based on no. of satellites used in Fastloc-GPS position calculation (1-6 for 4,5,6,7,8,>8 sats, resp.)
%  sigmaFall: Scale parameter for FGPS error distributions in km, for q=1,..,6 and dimension=x,y. Default is [24.48,34.35; 19.22,25.47; 13.13,17.24; 10.69,14.10; 9.20,11.22; 8.01,9.71]/1000 based on field tests
%  nuFall: Shape parameter for FGPS error distributions, for q=1,..,6 and dimension=x,y. Default is [0.92,1.09; 1.44,1.63; 2.50,2.73; 3.76,4.82; 5.19,5.12; 5.99,5.85] based on field tests
%  type: type of position fix for first location of the animal (1: FGPS fix, 2: visual fix)
%  Xw: position of animal based on visual observation.
%  tsurf: maximum duration of a surfacing period in sec. Used for replacing timestamps of fixes when 2 or more fixes occur during the same surfacing. Default is 5 s
%  nburnin: total number of burn-in samples. Default is 200k
%  nsamples: number of MCMC samples to keep (=total/thin). Default is 16k
%  thin: thinning factor for MCMC chains. Default is 5
%  initsigma: initial values of MCMC chains for sigma (rows: chain 1,2; columns: dimension x,y). Change when the chains don't converge. Default is [0.008,0.008; 0.015,0.015]
%  workingDir: JAGS working directory for storing large temporary files. Change when fitting multiple models in parallel. Default is 'C:\Temp\jagstmp' 
%  doParallel: Run the 2 chains on different cores? (0: no, 1: yes). Requires Matlab Parallel Computing Toolbox. Default is no
%  plots: Create data plots? (0: no, 1: yes). Plots the 2D track, velocity correction timeseries, and posterior distributions and trace plots for sigma. Default is yes 
%
% Note(1)-All time vectors are in seconds relative to the time of the first position fix on the animal
% Note(2)-All position vectors are in Carteian coordinates in km relative to the coordinates of the first position fix on the animal
% Note(3)-Input data as column vectors
%
% Requirements: matjags.m, JAGS, directory with JAGS executable in windows path
% Ref [1]. Wensveen PJ, Thomas L, Miller PJO 2015. A path reconstruction method integrating dead-reckoning and position fixes applied to humpback whales. Movement Ecology 3:31

%% Check input and set default values
narginchk(22,22);

%  Xb: position of observation boat at times tb 
%  Phi: bearing to animal (boat-animal angle relative to N) in rad spanning (-pi,pi] at times tb
%  rho: wrapped Cauchy concentration parameter for bearing error distribution. Default is 0.897326 based on field tests
%  R: range from observation boat to animal in km at times tb 

% Set default values for empty arguments
if isempty(s), s=1; end
if isempty(sigmaw), sigmaw=[0.075,0.025]; end
if isempty(sigmaFall), sigmaFall=[24.48,34.35; 19.22,25.47; 13.13,17.24; 10.69,14.10; 9.20,11.22; 8.01,9.71]/1000; end
if isempty(nuFall), nuFall=[0.92,1.09; 1.44,1.63; 2.50,2.73; 3.76,4.82; 5.19,5.12; 5.99,5.85]; end
if isempty(tsurf),tsurf=5; end
if isempty(nburnin), nburnin=2e5; end
if isempty(nsamples), nsamples=16e3; end
if isempty(thin), thin=5; end
if isempty(initsigma), initsigma=[0.008,0.008; 0.015,0.015]; end
if isempty(workingDir), workingDir='C:\Temp\jagstmp'; end
if isempty(doParallel), doParallel=0; end
if isempty(plots), plots=1; end

% workingDir should not have backslash
if strcmp(workingDir(end),'\')
    workingDir = workingDir(1:end-1);
end

%% Pre-process data

% Replace timestamp FGPS fix with timestamp FGPS fix if they come from
% the same surfacing (this code works for up to 2 FGPS loggers)
ind = find(diff(tF)>0 & diff(tF)<=tsurf) + 1;
tF(ind) = tF(ind-1); 

% Replace timestamp visual fix with timestamp FGPS fix if they come from
% the same surfacing 
for i=1:length(tF)
    ind = find(((abs(tF(i)-tb))>0) & (abs(tF(i)-tb)<=tsurf));
    if ~isempty(ind)
        tb(ind) = tF(i);
    end
end
tseg = unique([tb; tF]); % start time of track segment j
delta = diff(tseg); % duration of track segment in s (delta_j)
N = length(tseg); % no. of track segments (= no. of positions to estimate in JAGS)

% Indices for nested indexing in observation models
Nb = length(tb);
ind = nan(Nb,1);
for i=1:Nb
    ind(i) = find(tseg == tb(i));
end
nbin = hist(ind(:,1),1:N); % no. of observations per segment
idxb = [1; cumsum(nbin')+1]; % observation indices for visual sightings 

NF = length(tF);
ind = nan(NF,1);
for i=1:NF
    ind(i) = find(tseg == tF(i));
end
nbin = hist(ind(:,1),1:N); % no. of observations per segment
idxF = [1; cumsum(nbin')+1]; % observation indices for FGPS

% Displacement per track segment from dead-reckoning
vx = s .* cos(p) .* sin(h); % velocity from dead-reckoning in m/s
vy = s .* cos(p) .* cos(h); 
for i=1:N-1
    ind = (t>=tseg(i)) & (t<tseg(i+1));
    ddrx(i,1) = sum(vx(ind)) / 1000; % displacement in km
    ddry(i,1) = sum(vy(ind)) / 1000;
end

% Assign FGPS error parameters to data based on no. of satellites used
q(q>6) = 6; % 9 sats and higher are combined in one category
tauF(:,1) = sigmaFall(q,1).^(-2); % precision, x
tauF(:,2) = sigmaFall(q,2).^(-2); % precision, y
nuF(:,1) = nuFall(q,1); % shape, x
nuF(:,2) = nuFall(q,2); % shape, y


%% Fit model
% Create data stucture "o" to send to JAGS
o.N = N;
o.ddrx = ddrx;
o.ddry = ddry;
o.delta = delta;
o.type = type;
o.idxF = idxF;
o.tauF = tauF;
o.nuF = nuF;
o.XFx = XF(:,1);
o.XFy = XF(:,2);
o.idxb = idxb;
o.Xwx = Xw(:,1);
o.Xwy = Xw(:,2);
o.m = m;
o.sigmaw = sigmaw;

nchains  = 2; % no. of mcmc chains
init0(1) = struct('sigma',initsigma(1,:),'vcor',0.1*ones(N-1,2)); % initial values chain 1
init0(2) = struct('sigma',initsigma(2,:),'vcor',-0.1*ones(N-1,2)); % initial values chain 2

% Sample using parallelization? (this requires Parallel Computing Toolbox)
if doParallel==1
    if matlabpool('size') == 0
        matlabpool open 2; % initialize 2 local workers
    end
end

% Run JAGS
fprintf( 'Running JAGS...\n' );
[samples, stats] = matjags( ...
    o, ...                              % Observed data   
    [cd, '\jags_model_v5.txt'], ... % File that contains model definition (UPDATED MARCH 2022)
    init0, ...                          % Initial values for latent variables
    'doparallel' , doParallel, ...      % Parallelization flag
    'nchains', nchains,...              % Number of MCMC chains
    'nburnin', nburnin,...              % Number of burnin steps
    'nsamples', nsamples, ...           % Number of samples to extract
    'thin', thin, ...                   % Thinning parameter
    'dic', 0, ...                       % DIC module off
    'monitorparams', {'x','sigma','vcor'}, ...  % List of latent variables to monitor
    'savejagsoutput' , 1 , ...          % Save command line output produced by JAGS?
    'verbosity' , 1 , ...               % 0=do not produce any output; 1=minimal text output; 2=maximum text output
    'workingDir', workingDir,...        % change this when running multiple copies of matlab
    'cleanup' , 0);                     % clean up of temporary files?

%% Post-process output data

% posterior mean track in relative Cartesian coordinates
ind = 1:10:nsamples; % keep 10% of posterior samples
xsamp = [squeeze(samples.x(1,ind,:,1)); squeeze(samples.x(2,ind,:,1))]';
ysamp = [squeeze(samples.x(1,ind,:,2)); squeeze(samples.x(2,ind,:,2))]';
vcxsamp = [squeeze(samples.vcor(1,ind,:,1)); squeeze(samples.vcor(2,ind,:,1))]';
vcysamp = [squeeze(samples.vcor(1,ind,:,2)); squeeze(samples.vcor(2,ind,:,2))]';
Nit = size(vcxsamp,2); % no. of iterations
Lt = length(t); % track length in samples
xall = nan(Lt,Nit,2); 
xall(1,:,1) = xsamp(1,:); 
xall(1,:,2) = ysamp(1,:);
for i=1:N-1 % segment loop
    ind = find((t>=tseg(i)) & (t<(tseg(i+1)))==1);
    for j = 1:Nit % posterior track loop
        dx = cumsum((vx(ind)+vcxsamp(i,j))) / 1000;
        dy = cumsum((vy(ind)+vcysamp(i,j))) / 1000;
        xall(1+ind,j,1) = xsamp(i,j) + dx; % posterior sample tracks, x
        xall(1+ind,j,2) = ysamp(i,j) + dy; % posterior sample tracks, y
    end
end
xmean = squeeze(mean(xall,2)); 

%% Create data plots
if plots==1
    
% 2D track in relative Cartesian coordinates
figure; plot(xmean(:,1), xmean(:,2), 'k-','Linewidth',2);
hold on; plot(XF(:,1),XF(:,2),'o','MarkerFaceColor','r','MarkerEdgeColor','none','Linewidth',0.5);
if ~isempty(Xw)
    ilrf = logical(m-1);
    plot(Xw(~ilrf,1), Xw(~ilrf,2), '^','MarkerFaceColor','g','MarkerEdgeColor','none','Linewidth',0.5);
    plot(Xw(ilrf,1), Xw(ilrf,2), 'v','MarkerFaceColor','m','MarkerEdgeColor','none','Linewidth',0.5);
end
hold off
legend('Most probable track','Fastloc-GPS position fix','Visual position fix','Visual position fix (LRF)')
xlabel('Easting (km)'); ylabel('Northing (km)');
set(gca,'DataAspectRatio',[1 1 1])

% Velocity correction, posterior mean and 95% CI
vclow = stats.ci_low.vcor';
vchigh = stats.ci_high.vcor';
vclow = [vclow(1:(N-1),1),vclow(N:end,1)]; % due to matjags bug
vchigh = [vchigh(1:(N-1),1),vchigh(N:end,1)]; % due to matjags bug
ttmp = t(1)+tseg; t2 = nan(length(ttmp)*2,1);
t2(1:2:end)=ttmp; t2(2:2:end)=ttmp; t2=t2(2:end-1);
vcor = stats.mean.vcor;
vc2 = nan((N-1)*2,2); vc2(1:2:end,:)=vcor; vc2(2:2:end,:)=vcor;
vclow2 = nan(length(vclow)*2,2); vclow2(1:2:end,:)=vclow; vclow2(2:2:end,:)=vclow;
vchigh2 = nan(length(vchigh)*2,2); vchigh2(1:2:end,:)=vchigh; vchigh2(2:2:end,:)=vchigh;

figure; plot(t2, vc2(:,1), 'k-', t2, vc2(:,2), 'g-','LineWidth',1.5)
l=legend('x','y');
hold on; plot(t2, vclow2(:,1), t2, vchigh2(:,1),...
    t2, vclow2(:,2), t2, vchigh2(:,2),'LineStyle','-','Color',[0.5 0.5 0.5]);
plot(t2, vc2(:,1), 'k-', t2, vc2(:,2),'g-', 'LineWidth',1.5); hold off;
set(gca,'XLim',t([1 end]),'YLim',[-2 2])
ylabel({'Velocity'; 'correction (m/s)'}); 
xlabel('Time (seconds since first fix)')

% posterior densities and trace plots for sigma
ind = 1:10:nsamples; % use 10% of posterior samples
bins = 0:0.001:0.1;
figure
for i=1:2
    ksdens1 = ksdensity(samples.sigma(1,ind,i),bins); % chain 1
    ksdens2 = ksdensity(samples.sigma(2,ind,i),bins); % chain 2
    
    subplot(2,2,i); plot(bins,ksdens1,'k',bins,ksdens2,'g');
    hold on; plot(0,0,'Marker','none','LineStyle','none'); hold off
    set(gca,'XLim',[0 0.1])
    legend(['\mu = ' num2str(stats.mean.sigma(i))],...
        ['sd = ' num2str(stats.std.sigma(i))],...
        ['R_h_a_t = ' num2str(stats.Rhat.sigma(i))])
    dstr='xy'; title(['\sigma_' dstr(i)])
    xlabel('Velocity (m/s)'); ylabel('Density')
    
    subplot(2,2,i+2); plot(ind,samples.sigma(1,ind,i)','k-',ind,samples.sigma(2,ind,i)','g-')
    ylabel('Velocity (m/s)'); xlabel('Sample no.')
end

end

% Nested function to write JAGS model was removed

end