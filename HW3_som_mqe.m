%% Main Script for Rotor-Bearing Health Assessment using SOM and SOM-MQE
clc; clear; close all;
%% Data Loading and Preparation
% Define file paths
trainHealthyPath = './Training/Healthy';
trainFaulty1Path = './Training/Faulty/Unbalance 1';
trainFaulty2Path = './Training/Faulty/Unbalance 2';
testPath = './Testing';
% Load training data
[trainData, trainLabels] = loadData({trainHealthyPath, trainFaulty1Path, trainFaulty2Path}, [1, 2, 3]);
% Load testing data
[testData, ~] = loadData({testPath}, []);
%% Feature Extraction
fs = 2560; % Sampling frequency
rotFreq = 20; % Rotational frequency
% Extract features for training data
trainFeatures = [];
for i = 1:length(trainData)
[timeFeatures, freqFeatures] = extractFeatures(trainData{i}, fs, rotFreq);
trainFeatures(i,:) = [timeFeatures, freqFeatures];
end
% Extract features for testing data
testFeatures = [];
for i = 1:length(testData)
[timeFeatures, freqFeatures] = extractFeatures(testData{i}, fs, rotFreq);
testFeatures(i,:) = [timeFeatures, freqFeatures];
end
% Feature normalization
[~, mu, sigma] = zscore(trainFeatures);
trainFeaturesNorm = (trainFeatures - mu)./sigma;
testFeaturesNorm = (testFeatures - mu)./sigma;
%% SOM Training and Visualization (Multi-class Approach)
% Create and train SOM with larger grid
somSize = [8 8]; % Increased from [5 5]
net = selforgmap(somSize);
net = train(net, trainFeaturesNorm');
% Assign labels to SOM neurons
somNeurons = vec2ind(net(trainFeaturesNorm'));
neuronLabels = zeros(1, prod(somSize));
for i = 1:prod(somSize)
neuronSamples = find(somNeurons == i);
if ~isempty(neuronSamples)
neuronLabels(i) = mode(trainLabels(neuronSamples));
end
end
% Predict training labels
trainPredicted = neuronLabels(somNeurons);
% Confusion matrix for training data
figure;
plotconfusion(categorical(trainLabels), categorical(trainPredicted'), 'SOM Training Confusion Matrix');
%% SOM-MQE Approach (Multi-model)
% Train individual SOMs for each class with larger grids
classModels = cell(1,3);
for c = 1:3
classData = trainFeaturesNorm(trainLabels == c,:)';
classNet = selforgmap([4 4]); % Increased from [3 3]
classNet = train(classNet, classData);
classModels{c} = classNet;
end
% Calculate MQE for test data
testMQE = zeros(size(testFeaturesNorm,1), 3);
for c = 1:3
[~, mqe] = getMQE(classModels{c}, testFeaturesNorm');
testMQE(:,c) = mqe;
end
% Assign labels based on minimum MQE
[~, testPredictedMQE] = min(testMQE, [], 2);
%% Results Visualization
% SOM Hits Plot with hexagons
figure;
plotsomhits(net, trainFeaturesNorm');
title('SOM Hits Diagram (8x8 Grid)');
% SOM Neighbor Distances with hexagons
figure;
plotsomnd(net);
title('SOM Neighbor Distances (8x8 Grid)');
% MQE Distribution
figure;
boxplot(testMQE, 'Labels', {'Healthy', 'Unbalance1', 'Unbalance2'});
ylabel('MQE Value');
title('MQE Distribution Across Classes');
% Display Test Predictions
fprintf('\nTest Sample Predictions (SOM-MQE):\n');
disp(array2table([(1:30)', testPredictedMQE], ...
'VariableNames', {'SampleID', 'PredictedClass'}));
%% Helper Functions
function [data, labels] = loadData(paths, classLabels)
data = {};
labels = [];
for p = 1:length(paths)
files = dir(fullfile(paths{p}, '*.txt'));
for i = 1:length(files)
filePath = fullfile(files(i).folder, files(i).name);
% Read data skipping first 5 lines
fileID = fopen(filePath, 'r');
textscan(fileID, '%s', 5, 'Delimiter', '\n');
sig = textscan(fileID, '%f');
fclose(fileID);
data{end+1} = sig{1};
if ~isempty(classLabels)
labels(end+1) = classLabels(p);
end
end
end
labels = labels(:);
end
function [timeFeatures, freqFeatures] = extractFeatures(signal, fs, rotFreq)
% Time-domain features
timeFeatures = [
rms(signal), ... % RMS
peak2peak(signal), ... % Peak-to-peak
std(signal), ... % Standard deviation
skewness(signal), ... % Skewness
kurtosis(signal), ... % Kurtosis
max(signal)/rms(signal) ... % Crest factor
];
% Frequency-domain features
n = length(signal);
f = (0:n-1)*(fs/n);
amp = abs(fft(signal))/n*2;
amp = amp(1:floor(n/2)+1);
f = f(1:floor(n/2)+1);
freqRange = 5; % ±5 Hz range
freqFeatures = zeros(1,3);
for k = 1:3
targetFreq = k*rotFreq;
idx = find(f >= (targetFreq - freqRange) & f <= (targetFreq + freqRange));
if ~isempty(idx)
freqFeatures(k) = max(amp(idx));
end
end
end
function [bmu, mqe] = getMQE(net, data)
w = net.IW{1};
bmu = zeros(1, size(data,2));
mqe = zeros(1, size(data,2));
for i = 1:size(data,2)
dist = sum((w - data(:,i)').^2, 2);
[mqe(i), bmu(i)] = min(dist);
end
end
