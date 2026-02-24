%% Main Script for SOM and SOM-MQE Analysis
clear; close all; clc;
%% Data Loading with Correct Label Parsing
% Define paths - using relative paths from workspace root
train_healthy_path = fullfile(pwd, 'Training', 'Healthy');
train_faulty_path = fullfile(pwd, 'Training', 'Faulty');
test_path = fullfile(pwd, 'Testing');

% Check if paths exist
fprintf('Checking data paths...\n');
fprintf('Healthy path: %s (exists: %d)\n', train_healthy_path, isfolder(train_healthy_path));
fprintf('Faulty path: %s (exists: %d)\n', train_faulty_path, isfolder(train_faulty_path));
fprintf('Test path: %s (exists: %d)\n', test_path, isfolder(test_path));

% Load data with proper label parsing
[healthy_signals, healthy_labels] = load_data(train_healthy_path, 1);
[faulty_signals, faulty_labels] = load_data(train_faulty_path, 2);
[test_signals, test_labels] = load_data(test_path, 0);

% Combine training data
all_signals = [healthy_signals; faulty_signals];
all_labels = [healthy_labels; faulty_labels];
%% Enhanced Feature Extraction
fs = 2560; % Sampling frequency
rot_freq = 20; % Rotating frequency
freq_range = 5; % Frequency range ±5 Hz
% Initialize feature matrices
features_train = zeros(length(all_signals), 13);
features_test = zeros(length(test_signals), 13);
% Feature extraction for training data
for i = 1:length(all_signals)
    [t, f, amp] = FFTAnalysis(all_signals{i}, fs);
    features_train(i,:) = extract_features(all_signals{i}, t, f, amp, rot_freq, freq_range);
end
% Feature extraction for test data
for i = 1:length(test_signals)
    [t, f, amp] = FFTAnalysis(test_signals{i}, fs);
    features_test(i,:) = extract_features(test_signals{i}, t, f, amp, rot_freq, freq_range);
end
% Feature names
feat_names = {'RMS', 'Peak2Peak', 'Skewness', 'Kurtosis', 'Std', 'Mean', ...
              'Crest', 'Impulse', 'Clearance', 'Shape', '1x', '2x', '3x'};
%% Feature Normalization
[features_train_norm, mu, sigma] = zscore(features_train);
features_test_norm = (features_test - mu) ./ sigma;
%% SOM Training and Visualization
som_dim = [10 10];
net = selforgmap(som_dim);
net = train(net, features_train_norm');
%% Enhanced SOM Visualizations
% Component Planes with Feature Names
figure;
for i = 1:length(feat_names)
    subplot(4,4,i);
    plotsomplanes(net, i);
    title(feat_names{i}, 'Interpreter', 'none');
    colormap(jet);
end
sgtitle('SOM Component Planes with Feature Names');
% Weight positions
figure;
plotsompos(net, features_train_norm');
title({'SOM Weight Positions'; ['Features: ' strjoin(feat_names, ', ')]});
grid on;
% Neighbor distances
figure;
plotsomnd(net);
title({'SOM Neighbor Distances'; ['Features: ' strjoin(feat_names, ', ')]});
% Sample hits
figure;
plotsomhits(net, features_train_norm');
title({'SOM Sample Hits Distribution'; ['Features: ' strjoin(feat_names, ', ')]});
%% SOM Testing with Correct Label Handling
[~, bmu_indices] = vec2ind(net(features_train_norm'));
% Create label map for BMUs
num_neurons = prod(som_dim);
bmu_label_map = zeros(1, num_neurons);
for i = 1:num_neurons
    samples_in_neuron = find(bmu_indices == i);
    if ~isempty(samples_in_neuron)
        bmu_label_map(i) = mode(all_labels(samples_in_neuron));
    else
        bmu_label_map(i) = 0; % Handle empty neurons
    end
end
% Create predicted labels (FIXED DIMENSION)
predicted_labels = bmu_label_map(bmu_indices)';
% Remove invalid labels (0) and corresponding true labels
valid_indices = predicted_labels ~= 0;
all_labels_clean = all_labels(valid_indices);
predicted_labels_clean = predicted_labels(valid_indices);
% Confusion matrix (FIXED)
figure;
plotconfusion(categorical(all_labels_clean), categorical(predicted_labels_clean));
title({'SOM Classification Results'; ['Features: ' strjoin(feat_names, ', ')]});
%% SOM-MQE Analysis
% Train healthy model
healthy_features = features_train_norm(all_labels == 1, :);
net_healthy = selforgmap([3 3]);
net_healthy = train(net_healthy, healthy_features');
% Calculate MQE
[mqe_train, ~] = calculate_mqe(healthy_features, net_healthy);
[mqe_test, ~] = calculate_mqe(features_test_norm, net_healthy);
% Dynamic threshold
threshold = mean(mqe_train) + 2*std(mqe_train);
% MQE visualization
figure;
subplot(2,1,1);
plot(mqe_test, 'bo', 'DisplayName','Samples');
hold on;
plot([1 length(mqe_test)], [threshold threshold], 'r--', 'DisplayName','Threshold');
xlabel('Test Sample Index');
ylabel('MQE Value');
title({'MQE Values for Test Samples'; ['Features: ' strjoin(feat_names, ', ')]});
legend;
grid on;
subplot(2,1,2);
histogram(mqe_test, 20, 'FaceAlpha',0.7);
hold on;
xline(threshold, 'r--', 'LineWidth',2);
xlabel('MQE Values');
ylabel('Frequency');
title({'MQE Value Distribution'; ['Features: ' strjoin(feat_names, ', ')]});
grid on;
%% Supporting Functions
function [signals, labels] = load_data(folder_path, default_label)
    % default_label: optional parameter to set default label for all files (0 = auto-detect)
    if nargin < 2
        default_label = 0;  % Auto-detect from filename
    end
    
    if ~isfolder(folder_path)
        fprintf('Warning: Folder not found: %s\n', folder_path);
        signals = {};
        labels = [];
        return;
    end
    
    files = dir(fullfile(folder_path, '*.txt'));
    if isempty(files)
        fprintf('Warning: No .txt files found in: %s\n', folder_path);
        signals = {};
        labels = [];
        return;
    end
    
    signals = cell(length(files), 1);
    labels = zeros(length(files), 1);
    
    for i = 1:length(files)
        try
            data = dlmread(fullfile(folder_path, files(i).name), '', 5, 0);
            signals{i} = data;
            
            [~, name] = fileparts(files(i).name);
            if default_label > 0
                % Use provided label
                labels(i) = default_label;
            else
                % Auto-detect from filename
                if contains(name, 'Normal') || contains(name, 'Healthy')
                    labels(i) = 1;
                elseif contains(name, 'Unbalance 1') || contains(name, 'fault1')
                    labels(i) = 2;
                elseif contains(name, 'Unbalance 2') || contains(name, 'fault2')
                    labels(i) = 3;
                else
                    labels(i) = 0;  % Unknown label
                end
            end
        catch ME
            fprintf('Warning: Could not load file %s: %s\n', files(i).name, ME.message);
            signals{i} = [];
            labels(i) = 0;
        end
    end
    
    % Remove empty signals
    valid = ~cellfun(@isempty, signals);
    signals = signals(valid);
    labels = labels(valid);
    
    fprintf('Loaded %d files from: %s\n', length(signals), folder_path);
end
function [t, f, amp] = FFTAnalysis(x, fs)
    n = length(x);
    Y = fft(x);
    amp = abs(Y)/n * 2;
    amp = amp(1:floor(n/2)+1);
    f = (0:(floor(n/2))) * (fs/n);
    t = (0:n-1)/fs;
end
function features = extract_features(x, t, f, amp, rot_freq, freq_range)
    % Time-domain features
    rms_val = rms(x);
    p2p = max(x) - min(x);
    sk = skewness(x);
    kurt = kurtosis(x);
    std_val = std(x);
    mean_val = mean(x);
    crest = max(abs(x)) / rms_val;
    impulse = max(abs(x)) / mean(abs(x));
    clearance = max(abs(x)) / (mean(sqrt(abs(x))))^2;
    shape = rms_val / mean(abs(x));
    
    % Frequency-domain features
    idx_1x = find(f >= rot_freq - freq_range & f <= rot_freq + freq_range);
    idx_2x = find(f >= 2*rot_freq - freq_range & f <= 2*rot_freq + freq_range);
    idx_3x = find(f >= 3*rot_freq - freq_range & f <= 3*rot_freq + freq_range);
    
    amp_1x = max(amp(idx_1x));
    amp_2x = max(amp(idx_2x));
    amp_3x = max(amp(idx_3x));
    
    features = [rms_val, p2p, sk, kurt, std_val, mean_val, crest, ...
                impulse, clearance, shape, amp_1x, amp_2x, amp_3x];
end
function [mqe, bmu_indices] = calculate_mqe(data, net)
    outputs = net(data');
    [~, bmu_indices] = max(outputs);
    mqe = zeros(size(data,1), 1);
    
    for i = 1:size(data,1)
        mqe(i) = sum((data(i,:) - net.IW{1}(bmu_indices(i), :)).^2);
    end
end
