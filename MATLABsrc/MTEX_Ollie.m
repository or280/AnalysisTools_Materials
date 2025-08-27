clear all; close all
%IMPORTANT! Analysing orientations of different phases can be done for
%each phase seperately
%%
%Instructions for use of this code;

%1. Type in the command window "startup_mtex"
%2. Follow the instructions of the GUI to import the requried DATA
%3. Choose the EBSD function and add the .cpr file

%Original Code taken from "https://mtex-toolbox.github.io/EBSDImport.html",
%some instructions for use are available at this site


%% 
addpath('/Users/olliereed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - SuperelasticTitaniumAlloys/Useful/Scripts/Ollie Matlab/mtex-6.0.0')%this needs to be the directory 
startup_mtex
%where the MTex code is

CS = {... 
  'notIndexed',...
  crystalSymmetry('m-3m', [3.6 3.6 3.6], 'mineral', 'copper', 'color', [0.53 0.81 0.98])};

% plotting convention
setMTEXpref('xAxisDirection','east');
setMTEXpref('zAxisDirection','outOfPlane');

%% Specify File Names

% path to files
pname = '/Users/olliereed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - SuperelasticTitaniumAlloys/Ti2448/CR Ratio/SEM/Ti2448-03-24/Exports';

% which files to be imported
fname = [pname '/CR50.ctf'];

%% Import the Data

% create an EBSD variable containing the data
ebsd = EBSD.load(fname,CS,'interface','crc',...
  'convertEuler2SpatialReferenceFrame');


%% remove the non-indexed from the 'ebsd' variable
ebsd=ebsd('indexed');
% ebsd=B1;
% ebsd=B2;

%% Fill in the gaps - two options

F = halfQuadraticFilter; 
F.alpha = 0.25;
[grains,ebsd.grainId] = calcGrains(ebsd,'angle',5*degree);

ebsd = smooth(ebsd('indexed'),F,'fill'); %% This is a denoising + fill in the gaps. Longer time but better result. Must have done grain calculation first.

%ebsd = fill(ebsd('indexed'));  %fills based on nearest neighbours

%% Band contrast (Figure 1 - Greyscale EBSD map)
figure; plot(ebsd,ebsd.bc)
colormap gray % make the image grayscale
mtexColorbar

%% Plot the IPFZ (Figure 2 - Coloured EBSD map)

ipfKey.inversePoleFigureDirection = vector3d.X % Change the IPF orientation here

%colors = ipfKey.orientation2color(ebsd('copper').orientations);

plot(ebsd('copper'),colors,'micronbar','on','coordinates','off')
hold on %we need it on to overlay the grain boundaries
plot(grains.boundary,'lineWidth',1)
hold off

%% ipfZ Key (Figure 3 - produces the colour Key for the EBSD map @ Figure 2)
ipfKey = ipfColorKey(ebsd('copper'));
ipfKey.inversePoleFigureDirection = vector3d.X;

% this is the colored fundamental sector
figure; plot(ipfKey)
%% grain calculation (Figure 4 - Produces the Grain boundaries for Maps @ figure 1 and 2)
[grains,ebsd.grainId] = calcGrains(ebsd,'angle',5*degree); %you can change the angle
plot(grains.boundary,'lineWidth',1)
hold off %anything new will be ploted in a different figure

%% plot the IPDF Key (Figure 5 - the triangle with orientation distribution hits)
rr = vector3d.Y;rn = vector3d.Z;rt = vector3d.X
;  %set rolling normal and transverse directions
% plot the position of the z-Axis in crystal coordinates
%figure; plotIPDF(ebsd('copper').orientations,r,'MarkerSize',5,...
 %('MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)       For points/markers
 
figure('Position', [50, 50, 900, 1800]);  % Ensure enough figure space
% Manually define positions for the axes
ax1 = axes('Position', [0.05, 0.1, 0.28, 0.8]);  % First subplot position
plotIPDF(ebsd('copper').orientations, rr, 'smooth', 'colorrange', [0, 10]);
title('RD')

ax2 = axes('Position', [0.36, 0.1, 0.28, 0.8]);  % Second subplot position
plotIPDF(ebsd('copper').orientations, rn, 'smooth', 'colorrange', [0, 10]);
title('ND')

ax3 = axes('Position', [0.67, 0.1, 0.28, 0.8]);  % Third subplot position
plotIPDF(ebsd('copper').orientations, rt, 'smooth', 'colorrange', [0, 10]);
title('TD')

% Add a colorbar for all subplots (optional)
mtexColorbar;


%% Figure 6 - Orientation Distribution Function in (1 -1 0) Pole figures-for the colorful one-density 
% indication (ODF-orientation density function)
odf = calcDensity(ebsd('copper').orientations)
figure; plotPDF(odf,Miller(1,-1,0,odf.CS),'antipodal') %you can change the miller indices 
%and the 'complete' to 'upper' or 'lower'

caxis([0.08 1.5])

set(gca,'FontSize',20)

%% Figure 7 - Orientation Distribution Function in (1 0 0) Pole figure rotation (keeps the axes x-y unchanged)
rot = rotation.byAxisAngle(zvector,-10*degree);
odf_rotated = rotate(odf,rot);
figure; plotPDF(odf_rotated,Miller(1,0,0,odf.CS),'antipodal')

caxis([0.08 2.85])

set(gca,'FontSize',20)
%% Figure 8 - Orientation Distribution Function in (1 1 0) ODF for multiple planes
h = [Miller(1,1,0,odf.CS)];
%,Miller(1,-1,2,odf.CS),Miller(0,0,1,odf.CS)];
figure; plotPDF(odf,h,'antipodal','silent')
caxis([0 2.5])
%% Figure 9 - Orientation Distribution Function in (1 1 1) - Pole figures

h = Miller({0,0,1},ebsd('copper').CS); %you can change the miller indices
figure; plotPDF(ebsd('copper').orientations,h,'smooth', 'colorrange', [0, 10],'figSize','medium')

%% KAM
kam = ebsd.KAM / degree;
% lets plot it
figure; plot(ebsd,kam,'micronbar','on')
mtexColorbar
mtexColorMap LaboTeX
hold on
plot(grains.boundary,'lineWidth',1.5)
hold off
%%


%%
%% Dislocation density tensor

% after this process the 'ebsd' variable is converted to X*Y pixels
% variabe instead of pixels*1 

%redo grain calculation
[grains,ebsd.grainId] = calcGrains(ebsd,'angle',5*degree);

% smooth grain boundaries
grains = smooth(grains,5);

% a key the colorizes according to misorientation angle and axis
ipfKey = axisAngleColorKey(ebsd('indexed'));

% set the grain mean orientations as reference orinetations
ipfKey.oriRef = grains(ebsd('indexed').grainId).meanOrientation;

% denoise orientation data
F = halfQuadraticFilter;

ebsd = smooth(ebsd('indexed'),F,'fill',grains);

% plot the denoised data
ipfKey.oriRef = grains(ebsd('indexed').grainId).meanOrientation;

ebsdGrid = ebsd('indexed').gridify;

% compute the curvature tensor
kappa = ebsdGrid.curvature;

alpha = kappa.dislocationDensity;

dS = dislocationSystem.bcc(ebsdGrid.CS);

% size of the unit cell
a = norm(ebsdGrid.CS.aAxis);

% in bcc and fcc the norm of the burgers vector is sqrt(3)/2 * a
[norm(dS(1).b), norm(dS(end).b), sqrt(3)/2 * a];

dSRot = ebsdGrid.orientations * dS;

[rho,factor] = fitDislocationSystems(kappa,dSRot);

%%
figure
totalDensity=sum(abs(rho),2);
totalDensity=totalDensity.*factor;
% plot(ebsd,totalDensity)
plot(ebsd,totalDensity)
h=colorbar;
caxis([0 1e14])
h.Label.String='$\sum_{t=1}^{N}\left |\rho _{GND}^{t}   \right |$ ($m^{-2}$)'

%this part removes the unindexed rows from the totalDensity varible and
%creates a new one called indexedDensity
dataTable=[ebsd.phase totalDensity];
dataTable(dataTable(:, 1)== 0, :)= [];

indexedDensity=dataTable(:,2);

