import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

# Function to load data from a folder
def load_data(folder_path, label):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    data = []
    for file in files:
        # Skip header rows (e.g., rows containing 'Date')
        with open(file, 'r') as f:
            lines = f.readlines()
        # Find the first line with numeric data
        numeric_data = []
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            # Try to convert the first value in the line to a float
            try:
                float(line.split()[0])  # Check if the first column is numeric
                numeric_data.append(line)
            except (ValueError, IndexError):
                continue
        # Load numeric data only
        if numeric_data:
            signal = np.array([float(line.strip()) for line in numeric_data])
            data.append((signal, label))
    return data

# Function to compute FFT
def compute_fft(signal, fs=2560):
    n = len(signal)
    f = np.fft.fftfreq(n, d=1/fs)[:n//2]
    amp = np.abs(fft(signal))[:n//2] * 2 / n
    return f, amp

# Function to extract features
def extract_features(data):
    features = []
    for signal, label in data:
        # Time-domain features
        rms_val = np.sqrt(np.mean(signal**2))
        peak2peak = np.ptp(signal)
        std_dev = np.std(signal)
        skewness_val = skew(signal)
        kurtosis_val = kurtosis(signal)
        
        # Frequency-domain features
        f, amp = compute_fft(signal)
        rot_freq = 20  # Rotating frequency is 20Hz
        freq_range = 5  # Frequency range +/- 5Hz
        
        idx_1x = np.where((f >= rot_freq - freq_range) & (f <= rot_freq + freq_range))[0]
        idx_2x = np.where((f >= 2 * rot_freq - freq_range) & (f <= 2 * rot_freq + freq_range))[0]
        idx_3x = np.where((f >= 3 * rot_freq - freq_range) & (f <= 3 * rot_freq + freq_range))[0]
        
        amp_1x = np.max(amp[idx_1x]) if idx_1x.size > 0 else 0
        amp_2x = np.max(amp[idx_2x]) if idx_2x.size > 0 else 0
        amp_3x = np.max(amp[idx_3x]) if idx_3x.size > 0 else 0
        
        # Combine all features
        feature_vec = [rms_val, peak2peak, std_dev, skewness_val, kurtosis_val, amp_1x, amp_2x, amp_3x, label]
        features.append(feature_vec)
    return np.array(features)

# Function to calculate Fisher Score
def calculate_fisher_score(features, labels):
    unique_labels = np.unique(labels)
    num_features = features.shape[1]
    fisher_scores = np.zeros(num_features)
    
    for k in range(num_features):
        overall_mean = np.mean(features[:, k])
        numerator = 0
        denominator = 0
        
        for label in unique_labels:
            idx = labels == label
            class_mean = np.mean(features[idx, k])
            class_var = np.var(features[idx, k])
            numerator += np.sum(idx) * (class_mean - overall_mean)**2
            denominator += np.sum(idx) * class_var
        
        fisher_scores[k] = numerator / denominator if denominator != 0 else 0
    
    return fisher_scores

# Function to visualize Fisher Score
def visualize_fisher_score(fisher_scores, feature_names):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(fisher_scores)), fisher_scores, color='blue')
    plt.xticks(range(len(fisher_scores)), feature_names, rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Fisher Score')
    plt.title('Fisher Score for Features')
    plt.show()

# Function to train SOM and compute MQE
def train_som_and_compute_mqe(train_features, test_features, som_grid_size=(10, 10)):
    # Normalize features
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features[:, :-1])
    test_features_norm = scaler.transform(test_features[:, :-1])
    
    # Train SOM
    som = MiniSom(som_grid_size[0], som_grid_size[1], train_features_norm.shape[1], sigma=1.0, learning_rate=0.5)
    som.train_random(train_features_norm, 1000)
    
    # Plot SOM U-Matrix
    plt.figure(figsize=(8, 8))
    plt.pcolor(som.distance_map().T, cmap='bone_r')  # U-Matrix
    plt.colorbar()
    plt.title('SOM U-Matrix')
    plt.show()
    
    # Compute MQE for test data
    mqes = []
    for sample in test_features_norm:
        winner = som.winner(sample)
        bmu = som.get_weights()[winner]
        mqe = np.linalg.norm(bmu - sample)
        mqes.append(mqe)
    
    return mqes

# Main function
def main():
    # Use script directory for portable paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_healthy_path = os.path.join(script_dir, 'Training', 'Healthy')
    train_faulty1_path = os.path.join(script_dir, 'Training', 'Faulty', 'Unbalance 1')
    train_faulty2_path = os.path.join(script_dir, 'Training', 'Faulty', 'Unbalance 2')
    test_path = os.path.join(script_dir, 'Testing')
    
    # Load training and testing data
    train_healthy_data = load_data(train_healthy_path, 1)  # Label 1 for healthy
    train_faulty1_data = load_data(train_faulty1_path, 2)  # Label 2 for faulty1
    train_faulty2_data = load_data(train_faulty2_path, 3)  # Label 3 for faulty2
    test_data = load_data(test_path, 0)  # Label 0 for unknown (testing)
    
    # Combine all training data
    train_data = train_healthy_data + train_faulty1_data + train_faulty2_data
    
    # Extract features
    train_features = extract_features(train_data)
    test_features = extract_features(test_data)
    
    # Feature names
    feature_names = ['RMS', 'Peak2Peak', 'Std Dev', 'Skewness', 'Kurtosis', '1xRot', '2xRot', '3xRot']
    
    # Calculate Fisher Score
    fisher_scores = calculate_fisher_score(train_features[:, :-1], train_features[:, -1])
    
    # Visualize Fisher Score
    visualize_fisher_score(fisher_scores, feature_names)
    
    # Train SOM and compute MQE for test data
    mqes = train_som_and_compute_mqe(train_features, test_features)
    
    # Plot MQE results
    plt.figure(figsize=(10, 6))
    plt.plot(mqes, 'x-')
    plt.xlabel('Sample ID')
    plt.ylabel('MQE Value')
    plt.title('Minimum Quantization Error (MQE) for Test Data')
    plt.show()

if __name__ == '__main__':
    main()
