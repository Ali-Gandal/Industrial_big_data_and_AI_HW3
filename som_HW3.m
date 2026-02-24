function main()
    % Load data
    train_data = load_data('./Training');
    test_data = load_data('./Testing');

    % Sampling frequency
    fs = 2560;

    % Feature extraction for training data
    train_features = [];
    for i = 1:length(train_data)
        signal = train_data(i).signal;
        [time, freq, amp] = fft_analysis(signal, fs);
        features = feature_extraction(signal, freq, amp);
        train_features = [train_features; features];
    end

    % Feature extraction for testing data
    test_features = [];
    for i = 1:length(test_data)
        signal = test_data(i).signal;
        [time, freq, amp] = fft_analysis(signal, fs);
        features = feature_extraction(signal, freq, amp);
        test_features = [test_features; features];
    end

    % Normalize features
    [train_features, test_features] = normalize_features(train_features, test_features);

    % Check for NaN or Inf values in train_features
    if any(isnan(train_features(:))) || any(isinf(train_features(:)))
        error('Input data contains NaN or Inf values. Please check the data.');
    end

    % SOM Implementation
    som_dimensions = [10 10]; % SOM grid size
    som = selforgmap(som_dimensions);

    % Train the SOM
    som = train(som, train_features');

    % Visualize SOM topology and positions
    plotsomtop(som);
    plotsompos(som, train_features');

    % SOM-MQE for testing data
    mqe_results = [];
    for i = 1:size(test_features, 1)
        sample = test_features(i, :);
        bmu = find_bmu(som, sample');
        mqe = calculate_mqe(som, bmu, sample');
        mqe_results = [mqe_results; mqe];
    end

    % Plot MQE results
    figure;
    plot(mqe_results, 'x-');
    xlabel('Sample ID');
    ylabel('MQE');
    title('MQE for Test Samples');
end

function data = load_data(folder_path)
    % Load data from text files in the specified folder
    files = dir(fullfile(folder_path, '*.txt'));
    data = struct('signal', {}, 'label', {});

    for i = 1:length(files)
        file_path = fullfile(folder_path, files(i).name);
        
        % Read the file, skipping the first 5 lines (headers)
        fileID = fopen(file_path, 'r');
        signal = textscan(fileID, '%f', 'HeaderLines', 5);
        fclose(fileID);
        
        % Convert cell array to numeric array
        signal = signal{1};
        
        % Extract label from file name
        label = extract_label(files(i).name);
        
        % Store signal and label in struct
        data(i).signal = signal;
        data(i).label = label;
    end
end

function label = extract_label(file_name)
    % Extract label from file name
    if contains(file_name, 'Normal Data')
        label = 1; % Healthy
    elseif contains(file_name, 'Unbalance 2')
        label = 0; % Faulty
    else
        label = NaN; % Unknown
    end
end

function [time, freq, amp] = fft_analysis(signal, fs)
    % Perform FFT analysis
    n = length(signal);
    time = (0:n-1) / fs;
    freq = (0:n-1) * (fs / n);
    amp = abs(fft(signal)) / n * 2;
    amp = amp(1:floor(n/2));
    freq = freq(1:floor(n/2));
end

function features = feature_extraction(signal, freq, amp)
    % Extract features from the signal
    % Time-domain features
    rms_value = rms(signal);
    peak_to_peak = max(signal) - min(signal); % Custom peak-to-peak calculation
    std_dev = std(signal);
    skewness_value = skewness(signal);
    kurtosis_value = kurtosis(signal);

    % Frequency-domain features
    rot_freq = 20; % Rotating frequency is 20Hz
    freq_range = 5; % Frequency range +/- 5Hz

    idx_1x = find((freq >= rot_freq - freq_range) & (freq <= rot_freq + freq_range));
    idx_2x = find((freq >= 2 * rot_freq - freq_range) & (freq <= 2 * rot_freq + freq_range));
    idx_3x = find((freq >= 3 * rot_freq - freq_range) & (freq <= 3 * rot_freq + freq_range));

    amp_1x = max(amp(idx_1x));
    amp_2x = max(amp(idx_2x));
    amp_3x = max(amp(idx_3x));

    % Combine all features
    features = [rms_value, peak_to_peak, std_dev, skewness_value, kurtosis_value, amp_1x, amp_2x, amp_3x];
end

function [train_features, test_features] = normalize_features(train_features, test_features)
    % Normalize features using z-score normalization
    mean_train = mean(train_features);
    std_train = std(train_features);

    train_features = (train_features - mean_train) ./ std_train;
    test_features = (test_features - mean_train) ./ std_train;
end

function bmu = find_bmu(som, sample)
    % Find the Best Matching Unit (BMU) for a given sample
    weights = som.IW{1};
    distances = sqrt(sum((weights - sample').^2, 2));
    [~, bmu] = min(distances);
end

function mqe = calculate_mqe(som, bmu, sample)
    % Calculate Minimum Quantization Error (MQE)
    weights = som.IW{1};
    bmu_weights = weights(bmu, :);
    mqe = norm(bmu_weights' - sample);
end