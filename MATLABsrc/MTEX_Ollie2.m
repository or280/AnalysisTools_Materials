%% O. Reed 2025 Mtex scripts for plotting EBSD

%% Clear all variables from any previous sessions.

clear all; close all

% IMPORTANT! Analysing orientations of different phases must be done for
% each phase seperately, with the variables cleared inbetween


%% Start-up the Mtex library to use the functionalities

% Location of the M-tex installation folder
addpath('/Users/olliereed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - SuperelasticTitaniumAlloys/Useful/Scripts/Ollie Matlab/mtex-6.0.0')%this needs to be the directory 
startup_mtex


%% Specify crystal symmetry of phases involved and a plotting convention

% crystal symmetry
CS = {... 
  'notIndexed',...
  crystalSymmetry('m-3m', [3.2 3.2 3.2], 'mineral', 'Titanium cubic', 'color', [0.53 0.81 0.98])};

% plotting convention
how2plot = plottingConvention(zvector,xvector);


%% Specify File Names

% path to files
pname = '/Users/olliereed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - SuperelasticTitaniumAlloys/Ti2448/Large Grained/SEM';

% which files to be imported
fname = [pname '/TS04.cpr'];


%% Import the Data

% create an EBSD variable containing the data
ebsd = EBSD.load(fname,CS,'interface','crc',...
  'convertEuler2SpatialReferenceFrame');
ebsd.plottingConvention = how2plot;


%% Correct Data

% Correct for rotation of sample relative to aquisition
rot = rotation.byEuler(45*degree,0*degree,0*degree);
ebsd = rotate(ebsd,rot,'keepXY');


%% Set the EBSD to include only the indexed data
% Note that for multiple phases you will need to specify which phase you
% want to look at here

ebsd=ebsd('indexed')


%% Plot the data to see what it looks like

plot(ebsd)


%% Fill in the gaps in the data (the unindexed data)
% There are two options for this:

%% Nearest neighbour filling (quick)

ebsd = fill(ebsd);


%% Denoising and filling simultaneously (longer but better)

F = halfQuadraticFilter;
F.alpha = 0.25;
ebsd = smooth(ebsd,F,'fill');


%% Export the rotated, filled and denoised EBSD data to save this step next time

ebsd.export('/Users/olliereed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - SuperelasticTitaniumAlloys/Ti2448/CR Ratio/SEM/EBSD/MTex Exports/CR90_corr.ctf')


%% Calculate the grain boundaries 

[grains,ebsd.grainId] = calcGrains(ebsd,'angle',5*degree);
grains = smooth(grains,5);

%% Band contrast image. 

figure; plot(ebsd,ebsd.bc)
% make the image grayscale
colormap gray 
mtexColorbar


%% Plot ipf map

% this defines an ipf color key for the phase
ipfKey = ipfColorKey(ebsd);
% change this for the direction you are interested in
ipfKey.inversePoleFigureDirection = vector3d.Z; 
colors = ipfKey.orientation2color(ebsd.orientations);

plot(ebsd,colors,'micronbar','on','coordinates','off')

% To plot the ipf key
% plot(ipfKey)


%% Plot all 3 ipf map directions (with grain boundaries optional)

% this defines an ipf color key for the phase
ipfKey = ipfColorKey(ebsd);

% Define the figure, with the number of panes across and down
newMtexFigure('layout',[2,2]);

% First subplot, Y direction (RD)
nextAxis
ipfKey.inversePoleFigureDirection = vector3d.Y; 
colors = ipfKey.orientation2color(ebsd.orientations);
plot(ebsd,colors,'micronbar','on','coordinates','off')
hold on
plot(grains.boundary,'lineWidth',1.5)
hold off
title('RD')

% Second subplot, X direction (TD)
nextAxis
ipfKey.inversePoleFigureDirection = vector3d.X; 
colors = ipfKey.orientation2color(ebsd.orientations);
plot(ebsd,colors,'micronbar','on','coordinates','off')
hold on
plot(grains.boundary,'lineWidth',1.5)
hold off
title('TD')

% Third subplot, Z direction (ND)
nextAxis
ipfKey.inversePoleFigureDirection = vector3d.Z; 
colors = ipfKey.orientation2color(ebsd.orientations);
plot(ebsd,colors,'micronbar','on','coordinates','off')
hold on
plot(grains.boundary,'lineWidth',1.5)
hold off
title('ND')


%% Plot the grain boundaries onto a pre-existing figure

hold on
plot(grains.boundary,'lineWidth',1.5)
hold off


%% Calculate and plot the KAM map

ebsd = ebsd.gridify;
kam = ebsd.KAM('threshold',2.5*degree,'order',1) ./ degree;

% To append the plot onto the ipf maps then include the following line
nextAxis

plot(ebsd,kam,'micronbar','on')
setColorRange([0,2])
mtexColorbar
mtexColorMap LaboTeX

% To plot the grain boundaries as well
hold on
plot(grains.boundary,'lineWidth',1.5)
hold off
title("KAM")

%% Calculate and plot the Grain Ref Orientation Deviation (GROD) map

% Compute the grain reference orientation deviation (the difference of each
% pixel from the average orientation of a grain)
grod = ebsd.calcGROD(grains);

% To append the plot onto the ipf maps then include the following line
nextAxis

plot(ebsd,grod.angle./degree,'micronbar','off')
mtexColorbar('title',{'misorientation angle in degree'})
mtexColorMap LaboTeX

% To plot the grain boundaries as well
hold on
plot(grains.boundary,'lineWidth',1.5)
hold off
title("GROD")

%% Calculate and plot the Grain Orientation Spread (GOS) map

% compute the GOS (the spread in orientations per grain)
GOS = grainMean(ebsd, grod.angle, grains);

% To append the plot onto the ipf maps then include the following line
nextAxis

plot(grains, GOS ./ degree)
setColorRange([0,1])
mtexColorbar('title','GOS in degree')

% To plot the grain boundaries as well
hold on
plot(grains.boundary,'lineWidth',1.5)
hold off
title("GOS")


%% Plot the IPF triangles for all 3 directions

% Set the rolling directions
TD = vector3d.X; RD = vector3d.Y; ND = vector3d.Z;

% Define the figure, with the number of panes across and down
newMtexFigure('layout',[1,3]);

plotIPDF(ebsd.orientations, RD, 'smooth', 'colorrange', [0, 5]);
title('RD')

nextAxis
plotIPDF(ebsd.orientations, ND, 'smooth', 'colorrange', [0, 5]);
title('ND')

nextAxis
plotIPDF(ebsd.orientations, TD, 'smooth', 'colorrange', [0, 5]);
title('TD')

% Add a colorbar for all subplots (optional)
mtexColorbar;


%% Plot the pole figure circles for all 3 directions

% Set the crystal symmetry of the phase we are interested in
cs = ebsd("Titanium cubic").CS

% Set the directions of the pole figures
h = [Miller(1,0,0,cs),Miller(1,1,0,cs),Miller(1,1,1,cs)];

plotPDF(ebsd.orientations,h,'antipodal','projection','eangle','points',1000,'MarkerSize',4,'smooth','colorrange', [0, 5])

% Add a colorbar for all subplots (optional)
mtexColorbar;

%% Calculate the GND densities

ebsdgrid = ebsd.gridify;

% Compute the curvature tensor
kappa = ebsdgrid.curvature;

% The curvature is related to the dislocation density tensor
alpha = kappa.dislocationDensity;

% Define the dislocations which may be present
dS = dislocationSystem.bcc(ebsdgrid.CS);

% Size of the unit cell
a = norm(ebsdgrid.CS.aAxis);

% In bcc and fcc the norm of the burgers vector is sqrt(3)/2 * a
[norm(dS(1).b), norm(dS(end).b), sqrt(3)/2 * a];

% Energy of each dislocation type
dS(dS.isEdge).u = 1;
dS(dS.isScrew).u = 1 - 0.3;

% Rotate the dislocation tensor into the specimen reference frame
dSRot = ebsd.orientations * dS;

% Now we have all the dislocations that may exist, we need to fit them to
% the dislocation density tensor
[rho,factor] = fitDislocationSystems(kappa,dSRot);

% To get the total dislocation energies we multiply rho by the energies of 
% the dislocations and then by 10^16 (factor) to get to SI units.
GNDdensity = factor*sum(abs(rho .* dSRot.u),2);

%% Plotting the GND density

plot(ebsd,GNDdensity,'micronbar','on')
mtexColorMap('hot')
mtexColorbar
set(gca,'ColorScale','log');
set(gca,'CLim',[1e11 5e14]);
hold on
plot(grains.boundary,'linewidth',1.5)
hold off


%% Plotting the ipf triangles for ebsd for critical GOS values

% To find all grains with a GOS less than 1
threshold_angle = 0.25 * degree;

% Change sign to change high or low threshold
grain_indices = GOS < threshold_angle;
critGOS_grains = grains(grain_indices);

critGOS_grainids = critGOS_grains.id;
critGOS_ebsdindices = ismember(ebsd.grainId, critGOS_grainids);
critGOS_ebsd=ebsd(critGOS_ebsdindices);

% Set the rolling directions
TD = vector3d.X; RD = vector3d.Y; ND = vector3d.Z;

% Define the figure, with the number of panes across and down
newMtexFigure('layout',[1,3]);

plotIPDF(critGOS_ebsd.orientations, RD, 'smooth', 'colorrange', [0, 5]);
title('RD')

nextAxis
plotIPDF(critGOS_ebsd.orientations, ND, 'smooth', 'colorrange', [0, 5]);
title('ND')

nextAxis
plotIPDF(critGOS_ebsd.orientations, TD, 'smooth', 'colorrange', [0, 5]);
title('TD')

% Add a colorbar for all subplots (optional)
mtexColorbar;

%% Plotting the ipf triangles for ebsd for critical GND density values

% Threshold GND value
threshold_GND = 0.9e13;
critGND_pixels = GNDdensity < threshold_GND;
critGND_ebsd=ebsd(critGND_pixels);

% Set the rolling directions
TD = vector3d.X; RD = vector3d.Y; ND = vector3d.Z;

% Define the figure, with the number of panes across and down
newMtexFigure('layout',[1,3]);

plotIPDF(critGND_ebsd.orientations, RD, 'smooth', 'colorrange', [0, 5]);
title('RD')

nextAxis
plotIPDF(critGND_ebsd.orientations, ND, 'smooth', 'colorrange', [0, 5]);
title('ND')

nextAxis
plotIPDF(critGND_ebsd.orientations, TD, 'smooth', 'colorrange', [0, 5]);
title('TD')

% Add a colorbar for all subplots (optional)
mtexColorbar;

%% Save all variables to a file to avoid having to calculate GND densities again

save('/Users/olliereed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - SuperelasticTitaniumAlloys/Ti2448/Large Grained/SEM/TS04_var.mat')

%% Load in all variables if previously saved, so you may just plot things

load('/Users/olliereed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - SuperelasticTitaniumAlloys/Ti2448/CR Ratio/SEM/EBSD/MTex Exports/CR50_var.mat')