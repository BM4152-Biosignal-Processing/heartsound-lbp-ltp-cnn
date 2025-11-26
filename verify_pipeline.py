import numpy as np
from utils import Utils
import os

def test_pipeline():
    print("Testing Pipeline...")
    
    # 1. Generate Synthetic Signal
    sr = 2000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    
    print(f"Signal shape: {signal.shape}")
    
    # 2. Extract Features
    print("Extracting features...")
    features = Utils.extract_features(signal, sr, R=4, threshold=0.5, duration_sec=duration)
    print(f"Features shape: {features.shape}")
    
    # Expected length check
    target_len = int(sr * duration)
    expected_L = target_len - 2 * 4
    expected_features = 3 * expected_L
    
    if features.shape[0] == expected_features:
        print("Feature extraction shape matches expected.")
    else:
        print(f"Feature extraction shape mismatch. Expected {expected_features}, got {features.shape[0]}")
        
    # 3. Test ReliefF (Mock)
    print("Testing ReliefF (Mock)...")
    X = np.random.rand(10, 100) # 10 samples, 100 features
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    selected, weights = Utils.select_features_reliefF(X, y, n_selected_features=10, n_neighbors=3, n_iterations=5)
    print(f"Selected features: {selected}")
    print(f"Weights shape: {weights.shape}")
    
    if len(selected) == 10:
        print("ReliefF selection count correct.")
    else:
        print("ReliefF selection count incorrect.")

if __name__ == "__main__":
    test_pipeline()
