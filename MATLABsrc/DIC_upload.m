%% Load files into matlab

base_path = '/Users/olliereed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - SuperelasticTitaniumAlloys/Ti2448/Large Grained/Ti2448L1_ratchet1/';
fs = 2580;
fe = 2751;
diff = fe-fs+1;
data_cell = cell(1,diff);
dataset = [];

for i = fs:fe
    % Load a file
    filename = [base_path, sprintf('B%05d', i), '.txt'];

    % Need to replace the commas with decimal points

    % Read the entire file as a single string
    fileContent = fileread(filename); 
    % Replace commas with periods
    modifiedContent = strrep(fileContent, ',', '.'); 
    % Create a temporary file to store the modified data
    tempFilename = 'temp_data.txt'; 
    fileID = fopen(tempFilename, 'w');
    fprintf(fileID, '%s', modifiedContent);
    fclose(fileID);
    
    % Import the data from the temporary file
    data = readmatrix(tempFilename);

    % Remove rows with all zeros
    rows_to_keep = any(data,2); % Find rows with at least one non-zero element
    data = data(rows_to_keep,:);

    % Remove columns with all zeros
    cols_to_keep = any(data,1); % Find columns with at least one non-zero element
    data = data(:,cols_to_keep);

    % Store each data array in a cell
    adj = 1+i-fs;
    data_cell{adj} = data;
    
    if i == fs
        ref_size = size(data);  % Store the size of the first array
        dataset = zeros(ref_size(1), ref_size(2), diff); % Preallocate the 3D array
    elseif ~isequal(size(data), ref_size)
        warning(['Inconsistent data size in file B%05d. Dimensions are different from the first file.', i]);
        % Handle the inconsistency. Options:
        % 1. Skip this file:  continue;
        % 2. Pad/truncate:  You'd need to decide how to handle different sizes.
        error('Inconsistent data sizes.');
    end
    
    % Save the data to the 3D array
    dataset(:,:,adj) = data;

    % Delete the temporary file and variables
    
    clear data adj

end

delete(tempFilename);
clear fileID fileContent modifiedContent filename tempFilename ans
clear cols_to_keep rows_to_keep
clear data i

%% Import stress data and re-structure

% Import stress data and times
stressfile = [base_path, 'Ti2448L1_ratchet1_stress.txt'];
data = readmatrix(stressfile);
stress = data(:,1);
stress_time = data(:,2);
DIC_time = data(:,3);

% Remove any NaN values from the DIC_time wave
nan_indices = isnan(DIC_time);
DIC_time = DIC_time(~nan_indices);

% Interpolate to generate a stress wave with the same number of values as
% strain maps
stress = interp1(stress_time, stress, DIC_time);

clear i data stressfile nan_indices

%% Display a given image as a colour map

% Change this to the frame you wish to plot
plotdata = data_cell{30};

% Normalize the data
%data = (data - min(data(:))) / (max(data(:)) - min(data(:)));

% Display Image
imagesc(plotdata);

% Image settings
colormap(jet); 
colorbar;
axis equal;
xlabel('X');
ylabel('Y');
title('Array as Image');


%% Generate a film of the images

% Assuming your data is in a cell array called 'data_cell'
% where each cell contains one frame of your movie

% Create a figure window
figure

exportpath = '/Users/olliereed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - SuperelasticTitaniumAlloys/Ti2448/Large Grained/';
videofile = [exportpath + "mymovie.mp4"];

% Initialize the video writer
v = VideoWriter(videofile, 'MPEG-4'); % You can also use 'Motion JPEG AVI'
v.FrameRate = 15; % Set your desired frame rate
open(v);

% Loop through each frame
for i = 1:length(data_cell)
    % Display the current frame
    imagesc(data_cell{i});
    colormap(jet);
    colorbar;
    caxis([0 4]);
    axis equal;
    rounded_stress = round(stress(i+fs));
    title(['Frame: ' num2str(i) '   |   Stress = ' num2str(rounded_stress) ' MPa']);
    
    % Capture the current figure as a frame
    frame = getframe(gcf);
    
    % Write the frame to the videopwd

    writeVideo(v, frame);
end

% Close the video writer
close(v);

clear rounded_stress videofile


%% Plotting a stress strain curve of a region

% Define the region of interest (ROI):
roi_rows = 1:10;
roi_cols = 1:10;

% Calculate the average strain in each map
for i = 1:diff
    average_strain(i) = mean(dataset(roi_rows,roi_cols,i), 'all');
end

clear i 

% Plot the average strain against file number:
file_numbers = fs:fe; % Create a vector of file numbers
plot(average_strain, stress(file_numbers), '-o'); % '-o' for markers and lines
ylabel('Stress / MPa');
xlabel('Average Strain / %');
title('Average Strain vs. Stress');
grid on; % Add a grid for better readability
