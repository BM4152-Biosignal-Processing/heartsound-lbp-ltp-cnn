"""
Data preprocessing module for Heart Sound Classification
Implements 1D-LBP and 1D-LTP feature extraction as per:
"Heart sounds classification using CNN with 1D-LBP and 1D-LTP features"
Er, Mehmet Bilal - Applied Acoustics 2021
"""
import os
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
import librosa
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_DIR, CLASSES, CLASS_DIRS, SAMPLE_RATE, DURATION,
    BUTTER_ORDER, BUTTER_LOW, BUTTER_HIGH,
    LBP_NEIGHBORS, LTP_THRESHOLD, N_FEATURES_TO_SELECT, RANDOM_SEED
)


def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    """
    Apply Butterworth bandpass filter (from paper Section 3.1)
    Fifth-order Butterworth filter with band pass of 25-400 Hz
    
    Args:
        signal: Input signal
        lowcut: Low cutoff frequency (25 Hz)
        highcut: High cutoff frequency (400 Hz)
        fs: Sample rate
        order: Filter order (5)
    
    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure frequencies are within valid range
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def load_audio_file(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """
    Load and preprocess audio file
    
    Args:
        file_path: Path to the audio file
        sr: Target sample rate
        duration: Target duration in seconds
    
    Returns:
        Preprocessed audio signal
    """
    try:
        # Load audio file
        audio, orig_sr = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad or truncate to fixed length
        target_length = sr * duration
        
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Apply Butterworth bandpass filter (Section 3.1)
        audio = butter_bandpass_filter(audio, BUTTER_LOW, BUTTER_HIGH, sr, BUTTER_ORDER)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compute_1d_lbp(signal, P=LBP_NEIGHBORS):
    """
    Compute One Dimensional Local Binary Pattern (1D-LBP)
    As described in paper Section 3.3 and Equations (1), (2)
    
    For each sample, compare with P neighbors and create binary code.
    LBP(P,R) = sum(s(gi - gc) * 2^P) where s(x) = 1 if x >= 0 else 0
    
    Args:
        signal: Input 1D signal
        P: Number of neighbors (default 8)
    
    Returns:
        LBP feature histogram (256 bins for P=8)
    """
    n = len(signal)
    half_p = P // 2
    lbp_codes = []
    
    for i in range(half_p, n - half_p):
        gc = signal[i]  # Center value
        code = 0
        
        # Compare with P neighbors (4 on each side for P=8)
        neighbors_indices = list(range(i - half_p, i)) + list(range(i + 1, i + half_p + 1))
        
        for bit, j in enumerate(neighbors_indices):
            gi = signal[j]
            # s(gi - gc): 1 if gi >= gc, else 0
            if gi >= gc:
                code += (1 << bit)  # 2^bit
        
        lbp_codes.append(code)
    
    # Create histogram (2^P bins)
    n_bins = 2 ** P
    histogram, _ = np.histogram(lbp_codes, bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram
    histogram = histogram.astype(np.float32)
    histogram = histogram / (np.sum(histogram) + 1e-8)
    
    return histogram


def compute_1d_ltp(signal, P=LBP_NEIGHBORS, t=LTP_THRESHOLD):
    """
    Compute One Dimensional Local Ternary Pattern (1D-LTP)
    As described in paper Section 3.4, 3.5 and Equations (3)-(7)
    
    LTP uses 3 values: +1, 0, -1 based on threshold t
    s(qi, p, t) = +1 if qi >= (p + t)
                  0  if (p - t) < qi < (p + t)  
                 -1  if qi <= (p - t)
    
    Split into upper and lower patterns for reduced dimensionality.
    
    Args:
        signal: Input 1D signal
        P: Number of neighbors (default 8)
        t: Threshold value
    
    Returns:
        Tuple of (upper_histogram, lower_histogram)
    """
    n = len(signal)
    half_p = P // 2
    upper_codes = []
    lower_codes = []
    
    for i in range(half_p, n - half_p):
        p = signal[i]  # Center value
        upper_code = 0
        lower_code = 0
        
        # Compare with P neighbors
        neighbors_indices = list(range(i - half_p, i)) + list(range(i + 1, i + half_p + 1))
        
        for bit, j in enumerate(neighbors_indices):
            qi = signal[j]
            
            # Ternary function s(qi, p, t)
            if qi >= (p + t):
                # Upper pattern: keep +1, others 0
                upper_code += (1 << bit)
            elif qi <= (p - t):
                # Lower pattern: keep -1 as 1, others 0
                lower_code += (1 << bit)
            # else: both remain 0
        
        upper_codes.append(upper_code)
        lower_codes.append(lower_code)
    
    # Create histograms (2^P bins each)
    n_bins = 2 ** P
    upper_hist, _ = np.histogram(upper_codes, bins=n_bins, range=(0, n_bins))
    lower_hist, _ = np.histogram(lower_codes, bins=n_bins, range=(0, n_bins))
    
    # Normalize histograms
    upper_hist = upper_hist.astype(np.float32) / (np.sum(upper_hist) + 1e-8)
    lower_hist = lower_hist.astype(np.float32) / (np.sum(lower_hist) + 1e-8)
    
    return upper_hist, lower_hist


def extract_lbp_ltp_features(signal, P=LBP_NEIGHBORS, t=LTP_THRESHOLD):
    """
    Extract hybrid 1D-LBP and 1D-LTP features from signal
    As described in paper: combine LBP and LTP features
    
    Args:
        signal: Input audio signal
        P: Number of neighbors
        t: LTP threshold
    
    Returns:
        Combined feature vector (LBP + LTP_upper + LTP_lower)
    """
    # Compute 1D-LBP histogram (256 features for P=8)
    lbp_hist = compute_1d_lbp(signal, P)
    
    # Compute 1D-LTP histograms (256 + 256 features)
    ltp_upper, ltp_lower = compute_1d_ltp(signal, P, t)
    
    # Combine features: LBP + LTP_upper + LTP_lower = 768 features
    hybrid_features = np.concatenate([lbp_hist, ltp_upper, ltp_lower])
    
    return hybrid_features


def extract_multi_scale_features(signal, P=LBP_NEIGHBORS, t=LTP_THRESHOLD, n_segments=10):
    """
    Extract features from multiple segments of the signal
    This provides more temporal information
    
    Args:
        signal: Input audio signal
        P: Number of neighbors
        t: LTP threshold
        n_segments: Number of segments to divide signal
    
    Returns:
        Multi-scale feature vector
    """
    # Global features from entire signal
    global_features = extract_lbp_ltp_features(signal, P, t)
    
    # Segment-level features
    segment_length = len(signal) // n_segments
    segment_features = []
    
    for i in range(n_segments):
        start = i * segment_length
        end = start + segment_length
        segment = signal[start:end]
        
        if len(segment) > 2 * P:
            seg_feat = extract_lbp_ltp_features(segment, P, t)
            segment_features.append(seg_feat)
    
    if segment_features:
        # Statistical features from segments
        segment_features = np.array(segment_features)
        seg_mean = np.mean(segment_features, axis=0)
        seg_std = np.std(segment_features, axis=0)
        
        # Combine: global + segment statistics
        all_features = np.concatenate([global_features, seg_mean, seg_std])
    else:
        all_features = global_features
    
    return all_features


def relieff_feature_selection(X, y, n_features=N_FEATURES_TO_SELECT):
    """
    ReliefF-based feature selection (from paper Section 3.7)
    Selects the most discriminative features
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels
        n_features: Number of features to select
    
    Returns:
        Selected feature indices, feature importance scores
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_samples, n_feat = X.shape
    n_features = min(n_features, n_feat)
    
    # Initialize weights
    weights = np.zeros(n_feat)
    
    # Number of iterations
    m = min(100, n_samples)
    k = 5  # Number of nearest neighbors
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    np.random.seed(RANDOM_SEED)
    
    for _ in range(m):
        # Random instance
        idx = np.random.randint(n_samples)
        xi = X_scaled[idx]
        yi = y[idx]
        
        # Find nearest hits (same class)
        same_class_mask = y == yi
        same_class_indices = np.where(same_class_mask)[0]
        same_class_indices = same_class_indices[same_class_indices != idx]
        
        if len(same_class_indices) > 0:
            nn_same = NearestNeighbors(n_neighbors=min(k, len(same_class_indices)))
            nn_same.fit(X_scaled[same_class_indices])
            _, hit_indices = nn_same.kneighbors([xi])
            hits = X_scaled[same_class_indices[hit_indices[0]]]
            
            # Update weights based on hits
            for hit in hits:
                diff = np.abs(xi - hit)
                weights -= diff / (m * k)
        
        # Find nearest misses (different classes)
        for c in np.unique(y):
            if c == yi:
                continue
            
            class_mask = y == c
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) > 0:
                p_c = np.sum(class_mask) / n_samples
                
                nn_diff = NearestNeighbors(n_neighbors=min(k, len(class_indices)))
                nn_diff.fit(X_scaled[class_indices])
                _, miss_indices = nn_diff.kneighbors([xi])
                misses = X_scaled[class_indices[miss_indices[0]]]
                
                # Update weights based on misses
                for miss in misses:
                    diff = np.abs(xi - miss)
                    weights += (p_c / (1 - (np.sum(y == yi) / n_samples) + 1e-8)) * diff / (m * k)
    
    # Select top features
    selected_indices = np.argsort(weights)[-n_features:]
    
    return selected_indices, weights


def augment_audio(audio, sr=SAMPLE_RATE):
    """
    Apply data augmentation to audio signal
    
    Args:
        audio: Input audio signal
        sr: Sample rate
    
    Returns:
        List of augmented audio signals
    """
    augmented = []
    
    # 1. Add Gaussian noise
    for noise_level in [0.005, 0.01]:
        noise = np.random.normal(0, noise_level, len(audio))
        noisy = audio + noise
        noisy = noisy / (np.max(np.abs(noisy)) + 1e-8)
        augmented.append(noisy)
    
    # 2. Time stretching
    for rate in [0.9, 1.1]:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        # Adjust length
        target_len = len(audio)
        if len(stretched) < target_len:
            stretched = np.pad(stretched, (0, target_len - len(stretched)))
        else:
            stretched = stretched[:target_len]
        augmented.append(stretched)
    
    # 3. Amplitude scaling
    for scale in [0.8, 1.2]:
        scaled = audio * scale
        scaled = scaled / (np.max(np.abs(scaled)) + 1e-8)
        augmented.append(scaled)
    
    # 4. Time shifting
    for shift in [-1000, 1000]:
        shifted = np.roll(audio, shift)
        augmented.append(shifted)
    
    return augmented


def load_dataset(dataset_name='PASCAL', use_feature_selection=True, augment=True):
    """
    Load the complete dataset and extract LBP+LTP features
    
    Args:
        dataset_name: Name of the dataset ('PASCAL' or 'Physionet')
        use_feature_selection: Whether to apply ReliefF feature selection
        augment: Whether to apply data augmentation
    
    Returns:
        X: Feature matrix
        y: Labels
        selected_indices: Indices of selected features (if feature selection used)
    """
    from config import DATASET_CONFIG
    
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    config = DATASET_CONFIG[dataset_name]
    data_dir = config['path']
    classes = config['classes']
    class_dirs = config['class_dirs']
    
    X = []
    y = []
    
    print(f"Loading {dataset_name} dataset and extracting 1D-LBP + 1D-LTP features...")
    if augment:
        print("Data augmentation enabled (8x augmentation per sample)")
    
    for class_idx, class_name in enumerate(classes):
        # Handle Physionet's nested structure (train/val folders)
        if dataset_name == 'Physionet':
            # Search in both train and val folders
            target_dirs = []
            for subset in config['subsets']:
                subset_dir = os.path.join(data_dir, subset, class_dirs[class_name])
                if os.path.exists(subset_dir):
                    target_dirs.append(subset_dir)
        else:
            # Standard PASCAL structure
            target_dirs = [os.path.join(data_dir, class_dirs[class_name])]
            
        files_processed = 0
        for target_dir in target_dirs:
            if not os.path.exists(target_dir):
                print(f"Warning: Directory {target_dir} not found!")
                continue
            
            files = [f for f in os.listdir(target_dir) if f.endswith('.wav')]
            
            for file_name in files:
                file_path = os.path.join(target_dir, file_name)
                
                # Load and preprocess audio
                audio = load_audio_file(file_path)
                
                if audio is None:
                    continue
                
                # Extract LBP + LTP features for original audio
                features = extract_multi_scale_features(audio)
                X.append(features)
                y.append(class_idx)
                
                # Apply augmentation
                if augment:
                    augmented_audios = augment_audio(audio)
                    for aug_audio in augmented_audios:
                        # Apply filter again
                        aug_audio = butter_bandpass_filter(aug_audio, BUTTER_LOW, BUTTER_HIGH, 
                                                           SAMPLE_RATE, BUTTER_ORDER)
                        aug_features = extract_multi_scale_features(aug_audio)
                        X.append(aug_features)
                        y.append(class_idx)
                
                files_processed += 1
                
        print(f"Processed {files_processed} files for class '{class_name}'...")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset loaded: {X.shape[0]} samples")
    if len(X) > 0:
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Class distribution: {dict(zip(classes, np.bincount(y)))}")
        
        # Apply ReliefF feature selection
        selected_indices = None
        if use_feature_selection and X.shape[1] > N_FEATURES_TO_SELECT:
            print(f"\nApplying ReliefF feature selection...")
            selected_indices, weights = relieff_feature_selection(X, y, N_FEATURES_TO_SELECT)
            X = X[:, selected_indices]
            print(f"Selected {X.shape[1]} features")
            
        return X, y, selected_indices
    else:
        print("Error: No data loaded!")
        return np.array([]), np.array([]), None


def prepare_data(X, y, test_split=0.1, val_split=0.2):
    """
    Split data into train, validation, and test sets
    
    Args:
        X: Feature array
        y: Label array
        test_split: Fraction for test set
        val_split: Fraction for validation set
    
    Returns:
        Tuple of split data
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=RANDOM_SEED, stratify=y
    )
    
    # Second split: separate train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Add channel dimension for 1D-CNN
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    print(f"\nData split:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def get_kfold_splits(X, y, n_splits=10):
    """
    Get K-Fold cross validation splits (as in paper)
    
    Args:
        X: Feature array
        y: Label array
        n_splits: Number of folds
    
    Returns:
        Generator of (train_idx, val_idx) tuples
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    return skf.split(X, y)


if __name__ == "__main__":
    # Test feature extraction
    print("Testing 1D-LBP + 1D-LTP feature extraction...")
    
    # Load dataset
    X, y, selected_indices = load_dataset(use_feature_selection=True)
    
    print(f"\nFinal feature shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(X, y)
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
