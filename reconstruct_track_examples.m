clear all; clc; close all
% This script shows how to fit different versions of the track 
% reconstruction model described in [1]. Examples below use the data set of 
% humpback whale mn12_178 which can be found in Additional file 1 of [1], 
% available from http://ndownloader.figshare.com/files/7148801
%
% Model fitting may take several hours to complete for examples that 
% include Fastloc-GPS data
%
% Requirements: reconstruct_track(v1-v4).m, utm2deg.m, deg2utm.m, matjags.m, JAGS
%
% Ref [1]. Wensveen PJ, Thomas L, Miller PJO 2015. A path reconstruction method integrating dead-reckoning and position fixes applied to humpback whales. Movement Ecology 3:31

%% LOAD EXAMPLE DATASET

% DTAG-based data
data = xlsread('input_data.xlsx','DTAG','A3:J27087','basic');
tutc = data(:,1) + 693960; % UTC time
z = data(:,2); % depth in m
p = data(:,3); % pitch in rad
h = data(:,4); % heading in rad
s = data(:,5); % speed-through-water in m/s from flow noise
% Replace speed when depth<5m by linear interpolations
ind = ~isnan(s); tmp = (1:length(s))'; snonan = s(ind); 
s = interp1(tmp(ind), snonan, tmp, 'linear', mean(snonan)); 

% Visual tracking data
data = xlsread('input_data.xlsx','Visual','A3:I52','basic');
tutcb = data(:,1) + 693960; % UTC time
posb = data(:,2:3); % boat position [latitude, longitude]
Phi = (data(:,4)+data(:,5))*(pi/180); % absolute bearing re N in rad (=course-over-ground boat + relative bearing to whale) 
Phi = Phi - 2*pi*floor((Phi+pi)/(2*pi)); % wrap to pi
R = data(:,6)/1000; % range from boat to whale in km
m = data(:,7); % ranging method index (1: naked eye, 2: laser-range finder)
posw = data(:,8:9); % whale position [latitude, longitude]

% Fastloc GPS data
data = xlsread('input_data.xlsx','FGPS','A3:D161','basic');
tutcF = data(:,1) + 693960; % UTC time
q = data(:,2) - 3; % quality index (equal to no. of satellites minus 3)
posF = data(:,3:4); % gps position [latitude, longitude]
[tutcF, si] = sort(tutcF); % Place in chronological order (in case multiple loggers were used on the same animal)
q = q(si,:); posF = posF(si,:);

%% MAKE TIMESTAMPS AND POSITION COORDINATES RELATIVE TO FIRST POSITION FIX
t = round((tutc-tutc(1))*86400); % 1-s resolution vector based on DTAG data
tF = round((tutcF-tutc(1))*86400); % Timestamps of FGPS fixes
tb = round((tutcb-tutc(1))*86400); % Timestamps of visual fixes

% Type of first position fix
if tF(1)==0
    type = 1; % FGPS
elseif tb(1)==0
    type = 2; % visual
end
    
% Convert latlons to relative Cartesian coordinates (in km)
[Xbx, Xby, utmzone] = deg2utm(posb(:,1), posb(:,2)); % boat
[XFx, XFy] = deg2utm(posF(:,1), posF(:,2)); % FGPS
[Xwx, Xwy] = deg2utm(posw(:,1), posw(:,2)); % whale (visual fix)
utmzone = utmzone(1,:); % assumes positions fall within the same UTM zone!
if type==1 % reference position
    X0 = [XFx(1),XFy(1)];
elseif type==2
    X0 = [Xwx(1),Xwy(1)];
end
Xb = [Xbx-X0(1), Xby-X0(2)] / 1000;
XF = [XFx-X0(1), XFy-X0(2)] / 1000;
Xw = [Xwx-X0(1), Xwy-X0(2)] / 1000;

% Some more input arguments for the fitting functions
nburnin = 200000; % total number of burn-in samples 
nsamples = 16000; % number of posterior samples to store (=total/thin)
initsigma = [0.008,0.008; 0.015,0.015]; % initial values for sigma (rows: chain 1,2; columns: dimension x,y)
doParallel = 0; % Parallelization? (0: no; 1: yes) - requires Parallel Computing Toolbox

%% FIT MODELS

% Example 5 (NEW): Fit model to Fastloc GPS and visual position fixes
workingDir = 'C:\Temp\jagstmp\6';
sigmaw = [0.075,0.025];
[xmean,stats] = reconstruct_track_v5(t,p,h,s,tb,m,sigmaw,tF,XF,q,[],[],type,Xw,[],nburnin,nsamples,[],initsigma,workingDir,doParallel,[]);

% Example 1: Fit model to Fastloc GPS and visual observations of range and bearing 
workingDir = 'D:\Temp\jagstmp\1';
[xmean,stats] = reconstruct_track_v1(t,p,h,s,tb,Xb,Phi,[],R,m,[],tF,XF,q,[],[],type,Xw,[],nburnin,nsamples,[],initsigma,workingDir,doParallel,[]);

% Example 2.1: Fit model only to Fastloc GPS
workingDir = 'D:\Temp\jagstmp\2';
[xmean,stats] = reconstruct_track_v2(t,p,h,s,tF,XF,q,[],[],[],nburnin,nsamples,[],initsigma,workingDir,doParallel,[]);

% Example 2.2: Fit model only to Fastloc GPS, without forward speed data
workingDir = 'D:\Temp\jagstmp\3';
[xmean,stats] = reconstruct_track_v2(t,p,h,[],tF,XF,q,[],[],[],nburnin,nsamples,[],initsigma,workingDir,doParallel,[]);


% Adjust dataset for examples 3 and 4
X0 = [Xwx(1),Xwy(1)]; % adjust reference position
Xb = [Xbx-X0(1), Xby-X0(2)] / 1000;
Xw = [Xwx-X0(1), Xwy-X0(2)] / 1000;
t = t - tb(1); tb = tb - tb(1); % adjust reference time
ind = t>=0; t = t(ind); p = p(ind); h = h(ind); s = s(ind); % remove tag data before first visual fix 

% Example 3: Fit model only to visual observations of range and bearing 
workingDir = 'D:\Temp\jagstmp\4';
[xmean,stats] = reconstruct_track_v3(t,p,h,s,tb,Xb,Phi,[],R,m,[],Xw,nburnin,nsamples,[],initsigma,workingDir,doParallel,[]);

% Example 4: Fit model only to visual observations of position (for cases when range and bearing were not recorded)
workingDir = 'D:\Temp\jagstmp\5';
[xmean,stats] = reconstruct_track_v4(t,p,h,s,tb,Xw,m,[],nburnin,nsamples,[],initsigma,workingDir,doParallel,[]);


% plot posterior mean track in geographical coordinates
[pos(:,1),pos(:,2)] = utm2deg((xmean(:,1)*1000)+X0(1),(xmean(:,2)*1000)+X0(2),repmat(utmzone,length(t),1)); 
figure; plot(pos(:,2), pos(:,1), 'k-','Linewidth',2);
ylabel('Latitude (\circN)'); xlabel('Longitude (\circE)')
latlim = get(gca,'YLim'); lonlim = get(gca,'XLim');
latmean = ((latlim(1)+latlim(2))/2);
aspectratio = 1/cos(latmean*(pi/180));
set(gca,'DataAspectRatio',[aspectratio 1 1])

