import glob 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from minisom import MiniSom
from matplotlib.gridspec import GridSpec

def data_loading(paths, class_labels):
    alldata = pd.DataFrame(columns=['signal', 'label'])
    for idx, path in enumerate(paths):
        files = glob.glob(path + '/*.txt')
        
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
            
            signal = np.array(lines[5:]).astype(np.float32)
            label = class_labels[idx] if idx < len(class_labels) else -1
            # Store label as scalar integer
            temp_df = pd.DataFrame({'signal': [signal], 'label': [label]})
            alldata = pd.concat([alldata, temp_df], ignore_index=True)

    # Convert labels to integers
    alldata['label'] = alldata['label'].astype(int)
    return alldata
 
def feature_extraction(signal, fs=2560, rot_freq=20):
    # Time-domain features
    rms_val = np.sqrt(np.mean(signal**2))
    peak2peak = np.ptp(signal)
    std_dev = np.std(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    crest = np.max(np.abs(signal)) / rms_val
    
    # Frequency-domain features
    n = len(signal)
    freq = np.fft.fftfreq(n, d=1/fs)[:n//2]
    fft_vals = np.abs(np.fft.fft(signal))[:n//2] * 2 / n

    freq_features = []
    for k in [1, 2, 3]:
        target_freq = k * rot_freq
        mask = (freq >= target_freq - 5) & (freq <= target_freq + 5)
        if np.any(mask):
            freq_features.append(np.max(fft_vals[mask]))
        else:
            freq_features.append(0)

    return np.array([rms_val, peak2peak, std_dev, skewness, kurt, crest] + freq_features)
 
def train_som(data, som_size=(5,5), sigma=1.0, lr=0.5, iterations=1000):
    som = MiniSom(som_size[0], som_size[1], data.shape[1], sigma=sigma, learning_rate=lr)
    som.train_random(data, iterations)
    return som

def assign_labels(som, data, true_labels):
    winner_coordinates = np.array([som.winner(x) for x in data])
    label_map = np.full((som.get_weights().shape[0], som.get_weights().shape[1]), -1)
    for i in np.unique(winner_coordinates, axis=0):
        mask = np.all(winner_coordinates == i, axis=1)
        if np.any(mask):
            # Ensure labels are integers
            label_map[i[0], i[1]] = np.argmax(np.bincount(true_labels[mask].astype(int)))

    return label_map
 
def main():
    # Data paths
    train_paths = [
        './Training/Healthy',
        './Training/Faulty/Unbalance 1',
        './Training/Faulty/Unbalance 2'
    ]
    test_path = './Testing'
    
    # Load data with proper type conversion
    train_data = data_loading(train_paths, class_labels=[0, 1, 2])
    test_data = data_loading([test_path], class_labels=[-1])

    # Feature extraction
    train_features = np.array([feature_extraction(sig) for sig in train_data['signal']])
    test_features = np.array([feature_extraction(sig) for sig in test_data['signal']])

    # Feature normalization
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features)
    test_features_norm = scaler.transform(test_features)

    # SOM Training
    som = train_som(train_features_norm, som_size=(5, 5))

    # Assign labels with type-safe conversion
    label_map = assign_labels(som, train_features_norm, train_data['label'].values)

    # Training predictions
    train_predicted = np.array([label_map[som.winner(x)] for x in train_features_norm])

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(train_data['label'], train_predicted),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Unbalance1', 'Unbalance2'],
                yticklabels=['Healthy', 'Unbalance1', 'Unbalance2'])
    plt.title('SOM Training Confusion Matrix')
    plt.show()

    # SOM-MQE Analysis
    class_models = []
    for c in range(3):
        class_data = train_features_norm[train_data['label'] == c]
        class_som = train_som(class_data, som_size=(3, 3))
        class_models.append(class_som)

    test_mqe = np.zeros((len(test_features_norm), 3))
    for i, sample in enumerate(test_features_norm):
        for c in range(3):
            winner = class_models[c].winner(sample)
            test_mqe[i, c] = np.linalg.norm(sample - class_models[c]._weights[winner])

    test_predicted = np.argmin(test_mqe, axis=1)

    # Visualizations
    plt.figure(figsize=(10, 6))
    plt.boxplot([test_mqe[:,0], test_mqe[:,1], test_mqe[:,2]],
                labels=['Healthy', 'Unbalance1', 'Unbalance2'])
    plt.title('MQE Distribution Across Classes')
    plt.ylabel('MQE Value')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.title('SOM Neighbor Distances')
    plt.colorbar()

    plt.subplot(122)
    hits = np.zeros((5, 5))
    for x in train_features_norm:
        hits[som.winner(x)] += 1
    plt.pcolor(hits.T, cmap='Blues')
    plt.title('SOM Hit Histogram')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Test predictions
    print("\nTest Sample Predictions:")
    results = pd.DataFrame({
        'SampleID': range(1, 31),
        'PredictedClass': test_predicted
    })
    results['PredictedClass'] = results['PredictedClass'].map({
        0: 'Healthy', 
        1: 'Unbalance1', 
        2: 'Unbalance2'
    })
    print(results.to_string(index=False))
 
if __name__ == '__main__':
    main()
