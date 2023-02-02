function [xmean,stats] = reconstruct_track_v3(t,p,h,s,tb,Xb,Phi,rho,R,m,sigmar,Xw,nburnin,nsamples,thin,initsigma,workingDir,doParallel,plots)
% Track reconstruction from position fixes and dead-reckoning data using
% an approach based on [1]. Model fitting is performed via MCMC in JAGS.
%
% Version 3. Model structure with observation model for visual observations 
% of range and bearing
%
% Output arguments:
%  xmean: the most-probable track - posterior means of animal position x(i), in relative Cartesian coordinates
%  stats: data structure with posterior summaries and convergence parameter Rhat for parameters x (animal position), vcor (velocity correction), and sigma (velocity correction process error SDs)
%
% Input arguments:
%  t: time vector for 1 Hz tag data
%  p,h: pitch and heading of animal at times t
%  s: forward speed of animal at times t. Can be vector or, to assume constant speed, a scalar. Using a vector with speed estimates (e.g. based on flow noise) decreases runtime and improves convergence. Default is 1 m/s
%  tb: time of visual position fixes on the animal made from the observation boat
%  Xb: position of observation boat at times tb 
%  Phi: bearing to animal (boat-animal angle relative to N) in rad spanning (-pi,pi] at times tb
%  rho: wrapped Cauchy concentration parameter for bearing error distribution. Default is 0.897326 based on field tests
%  R: range from observation boat to animal in km at times tb 
%  m: ranging method index for each visual fix (1: naked eye, 2: laser-range finder)
%  sigmar: percent error SD in %, for range estimates by eye and laser-range finder. Default is [30.2, 10], based on field tests 
%  Xw: position of animal based on visual observation. Only used for plotting [OPTIONAL]
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
narginchk(19,19);

% Set default values for empty arguments
if isempty(s), s=1; end
if isempty(rho), rho=0.897326; end
if isempty(sigmar), sigmar=[30.2, 10]; end
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

tseg = unique(tb); % start time of track segment j
delta = diff(tseg); % duration of track segment in s (delta_j)
N = length(tseg); % no. of track segments (= no. of positions to estimate in JAGS)

% Note: no nested indexing for observation model here because each visual
% fix has an unique timestamp

% Displacement per track segment from dead-reckoning
vx = s .* cos(p) .* sin(h); % velocity from dead-reckoning in m/s
vy = s .* cos(p) .* cos(h); 
for i=1:N-1
    ind = (t>=tseg(i)) & (t<tseg(i+1));
    ddrx(i,1) = sum(vx(ind)) / 1000; % displacement in km
    ddry(i,1) = sum(vy(ind)) / 1000;
end


%% Fit model
% Create data stucture "o" to send to JAGS
o.N = N;
o.ddrx = ddrx;
o.ddry = ddry;
o.delta = delta;
o.Xbx = Xb(:,1);
o.Xby = Xb(:,2);
o.R = R;
o.m = m;
o.rho = rho;
o.sigmar = sigmar / 100;
o.Phi = Phi;
o.ones = ones(length(Phi),1); % for ones trick in JAGS

nchains  = 2; % no. of mcmc chains
init0(1) = struct('sigma',initsigma(1,:),'vcor',0.1*ones(N-1,2)); % initial values chain 1
init0(2) = struct('sigma',initsigma(2,:),'vcor',-0.1*ones(N-1,2)); % initial values chain 2

% Sample using parallelization? (this requires Parallel Computing Toolbox)
if doParallel==1
    if matlabpool('size') == 0
        matlabpool open 2; % initialize 2 local workers
    end
end

% Write model code to txt file
writeModel(workingDir);

% Run JAGS
fprintf( 'Running JAGS...\n' );
[samples, stats] = matjags( ...
    o, ...                              % Observed data   
    [workingDir '\' 'jags_model.txt'], ... % File that contains model definition
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
hold on;
if ~isempty(Xw)
    ilrf = logical(m-1);
    plot(Xw(~ilrf,1), Xw(~ilrf,2), '^','MarkerFaceColor','g','MarkerEdgeColor','none','Linewidth',0.5);
    plot(Xw(ilrf,1), Xw(ilrf,2), 'v','MarkerFaceColor','m','MarkerEdgeColor','none','Linewidth',0.5);
end
hold off
legend('Most probable track','Visual position fix','Visual position fix (LRF)')
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

%% Nested functions
    function writeModel(workingDir)
        modelstr = {'data {';'pi <- 3.141592653589';'pi2 <- 2*pi';'firstsd <- ifelse(m[1]-1, (sigmar[2]*R[1]), (sigmar[1]*R[1])) # sd of first location if visual   ';'}';[];...
            'model';'{';[];'## Priors';'sigma[1] ~ dunif(0, 0.1)  # Process noise sd, x';'sigma[2] ~ dunif(0, 0.1)  # Process noise sd, y';[];...
            '## variance-covariance precision matrix';'iSigma[1,1] <- sigma[1]^(-2)';'iSigma[1,2] <- 0';'iSigma[2,1] <- 0';'iSigma[2,2] <- sigma[2]^(-2)';[];...
            '## Priors on 1st velocity correction';'vcor[1,1] ~ dunif(-1, 1)';'vcor[1,2] ~ dunif(-1, 1)';[];...
            '## Priors on 1st position';'x[1,1] ~ dnorm(0, firstsd^(-2))';'x[1,2] ~ dnorm(0, firstsd^(-2))';[];...
            '## 2nd whale position = first position + displacement from DR + velocity correction*delta';'x[2,1] <- x[1,1] + ddrx[1] + vcor[1,1]/1000*delta[1]';'x[2,2] <- x[1,2] + ddry[1] + vcor[1,2]/1000*delta[1]';[];...
            '## CRW process on velocity correction';'for(j in 2:(N-1)){';'vcor[j,1:2] ~ dmnorm(vcor[j-1,], iSigma[,]/delta[j])';[];'## Update whale position  ';'x[j+1,1] <- x[j,1] + ddrx[j] + vcor[j,1]/1000*delta[j]';'x[j+1,2] <- x[j,2] + ddry[j] + vcor[j,2]/1000*delta[j]';'}';[];...
            '## Observation model for visual data';'for(j in 1:N){';'dbwx[j] <- x[j,1] - Xbx[j] # boat-whale orthogonal distance, x';'dbwy[j] <- x[j,2] - Xby[j] # boat-whale orthogonal distance, y';[];...
            '## Boat-whale range without error';'r[j] <- sqrt(dbwx[j]^2 + dbwy[j]^2) ';[];'## Boat-whale range with error depends upon measurement type';'R[j] ~ dnorm(r[j], rangesd[j]^(-2))';'rangesd[j] <- ifelse(m[j]-1, sigmar[2]*r[j], sigmar[1]*r[j])';[];...
            '## Bearing without error using four-quadrant arc tangent (-pi,pi]';'at[j] <- atan(dbwx[j] / dbwy[j])';'phi[j] <- ifelse(step(dbwy[j]), at[j], pi2*step(dbwx[j])+at[j]-pi)';[];...
            '## Ones trick for bearing with error [i.e. Phi[i] ~ wC(phi[j], rho)]';'wC[j] <- (1/pi2 * (1-rho^2) / (1+rho^2 - 2*rho*cos(Phi[j] - phi[j])))/1000';'ones[j] ~ dbern(wC[j])';'}';'}'};
        mkdir(workingDir)
        modelfullpath = [workingDir '\' 'jags_model.txt'];
        fileID = fopen(modelfullpath,'w');
        [nrows,ncols] = size(modelstr);
        for row = 1:nrows
            fprintf(fileID,'%s\r\n',modelstr{row,:}(:));
        end
        fclose(fileID);
    end


end