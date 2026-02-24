import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

def data_loading(filepath):
    files = glob.glob(filepath + '/**/*.txt', recursive=True)
    alldata = pd.DataFrame(columns=['signal', 'label'])

    # loop for each file
    for i, file in enumerate(files):
        print('Converting data file {}'.format(file))

        with open(file, 'r') as f:
            lines = f.readlines()

        t = np.array(lines[5:]).astype(np.float32)
        alldata.loc[i, 'signal'] = t

        filename = os.path.basename(file)
        label = ' '.join(filename.split(' ')[:2])
        if label == 'Normal Data':
            alldata.loc[i, 'label'] = 1
        elif label == 'Unbalance 2':
            alldata.loc[i, 'label'] = 0
        else:
            alldata.loc[i, 'label'] = np.nan

    return alldata

def feature_extraction(d):
    def rms(n):
        return np.sqrt(np.sum(np.square(n)) / len(n))

    x = d['signal']  # Access the 'signal' column
    f = d['frequency']  # Access the 'frequency' column
    amp = d['amplitude']  # Access the 'amplitude' column

    # Extracting statistical features: rms, p2p, skewness, kurtosis
    feavec = []
    feavec.append(rms(x))
    feavec.append(np.max(x) - np.min(x))
    feavec.append(skew(x))
    feavec.append(kurtosis(x))

    # Extracting frequency features: 1xRotating Frequency, 2xRotating Frequency, 3xRotating Frequency
    rot_freq = 20  # Rotating frequency is 20Hz
    freq_range = 5  # Frequency range +/- 5Hz

    idx_1x = np.where((f >= rot_freq - freq_range) & (f <= rot_freq + freq_range))
    idx_2x = np.where((f >= 2 * rot_freq - freq_range) & (f <= 2 * rot_freq + freq_range))
    idx_3x = np.where((f >= 3 * rot_freq - freq_range) & (f <= 3 * rot_freq + freq_range))

    feavec.append(np.max(amp[idx_1x]))
    feavec.append(np.max(amp[idx_2x]))
    feavec.append(np.max(amp[idx_3x]))

    feavec = np.array(feavec)
    feaname = ['rms', 'peak2peak', 'skewness', 'kurtosis', '1xrot', '2xrot', '3xrot']

    return feavec, feaname

def FFTAnalysis(x, fs):
    fnyq = fs / 2  # Nyquist Frequency

    # Generating the time axis
    dt = 1 / fs
    n = len(x)
    t = np.arange(0, n) * dt

    # Generating the frequency axis
    df = fs / n
    f = np.arange(0, n) * df

    # Applying the FFT Function
    amp = np.abs(np.fft.fft(x)) / n * 2
    amp = amp[f <= fnyq]
    f = f[f <= fnyq]

    return t, f, amp

def main():
    # Data Loading
    train_path = './Training'
    test_path = './Testing'
    train_data = data_loading(train_path)
    test_data = data_loading(test_path)

    # FFT Analysis on Training Data
    fs = 2560

    t_lst = []
    f_lst = []
    amp_lst = []

    for i in range(len(train_data)):
        x = train_data.signal.iloc[i]
        t, f, amp = FFTAnalysis(x, fs)

        t_lst.append(t)
        f_lst.append(f)
        amp_lst.append(amp)

    train_data['time'] = t_lst
    train_data['frequency'] = f_lst
    train_data['amplitude'] = amp_lst

    # Feature Extraction on Training Data
    feavec_lst = []
    for k in range(len(train_data)):
        dk = train_data.iloc[k]  # Get the row as a Series
        feavec, feaname = feature_extraction(dk)
        feavec_lst.append(feavec)
    train_data['feature'] = feavec_lst

    # Feature Normalization - Training Data
    feamat = np.array([x for x in train_data.feature])
    scaler = StandardScaler()
    feamat = scaler.fit_transform(feamat)

    # SOM Implementation
    som = MiniSom(10, 10, feamat.shape[1], sigma=1.0, learning_rate=0.5)
    som.train_random(feamat, 1000)  # Train the SOM

    # Visualize the SOM
    plt.figure(figsize=(10, 10))
    plt.pcolor(som.distance_map().T, cmap='bone_r')  # Distance map
    plt.colorbar()
    plt.show()

    # FFT Analysis on Test Data
    t_lst = []
    f_lst = []
    amp_lst = []
    for k in range(len(test_data)):
        x = test_data.signal.iloc[k]
        t, f, amp = FFTAnalysis(x, fs)

        t_lst.append(t)
        f_lst.append(f)
        amp_lst.append(amp)

    test_data['time'] = t_lst
    test_data['frequency'] = f_lst
    test_data['amplitude'] = amp_lst

    # Feature Extraction on Test Data
    test_feavec_lst = []
    for k in range(len(test_data)):
        dk = test_data.iloc[k]  # Get the row as a Series
        feavec, _ = feature_extraction(dk)
        test_feavec_lst.append(feavec)

    test_data['feature'] = test_feavec_lst
    test_feamat = np.array([x for x in test_data.feature])
    test_feamat = scaler.transform(test_feamat)

    # Calculate MQE for each test sample
    mqe_results = []
    for sample in test_feamat:
        bmu = som.winner(sample)  # Find the BMU
        bmu_weights = som.get_weights()[bmu]
        mqe = np.linalg.norm(sample - bmu_weights)  # Calculate MQE
        mqe_results.append(mqe)

    # Plot MQE results
    plt.plot(mqe_results, 'x-')
    plt.xlabel('Sample ID')
    plt.ylabel('MQE')
    plt.title('MQE for Test Samples')
    plt.show()

if __name__ == '__main__':
    main()
