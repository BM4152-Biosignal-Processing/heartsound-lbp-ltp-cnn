"""
Inference script for Heart Sound Classification
Uses 1D-LBP and 1D-LTP features as per Er, Mehmet Bilal (2021)

Usage:
    python predict.py <audio_file.wav>
    python predict.py <audio_file.wav> --model-path models/best_model.keras
    python predict.py --setup   # First-time setup to save scaler and feature indices
"""
import os
import numpy as np
import tensorflow as tf
import argparse
import pickle
import librosa
from sklearn.preprocessing import StandardScaler

from data_preprocessing import (
    butter_bandpass_filter, 
    extract_multi_scale_features,
    load_dataset
)
from config import (
    CLASSES, MODEL_SAVE_PATH, SAMPLE_RATE, DURATION,
    BUTTER_LOW, BUTTER_HIGH, BUTTER_ORDER, N_FEATURES_TO_SELECT
)


def load_audio_file(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Load and preprocess a single audio file"""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad or truncate to fixed length
        target_length = int(sr * duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def setup_inference():
    """
    Setup inference by loading training data to get scaler and feature indices.
    This should be run once after training to save the necessary files.
    """
    print("Setting up inference components...")
    print("Loading training data to extract feature indices and scaler...")
    
    # Load full dataset with feature selection AND augmentation (same as training)
    X, y, selected_indices = load_dataset(augment=True, use_feature_selection=True)
    
    # Fit scaler on full dataset
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Save scaler
    scaler_path = os.path.join(MODEL_SAVE_PATH, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")
    
    # Save feature indices
    if selected_indices is not None:
        indices_path = os.path.join(MODEL_SAVE_PATH, 'feature_indices.pkl')
        with open(indices_path, 'wb') as f:
            pickle.dump(selected_indices, f)
        print(f"Saved feature indices ({len(selected_indices)} features) to {indices_path}")
    
    print("\n✅ Setup complete! You can now use predict.py for inference.")


def load_model_and_components(model_path=None):
    """
    Load trained model, scaler, and feature indices
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Tuple of (model, scaler, feature_indices)
    """
    if model_path is None:
        model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.keras')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load scaler
    scaler_path = os.path.join(MODEL_SAVE_PATH, 'scaler.pkl')
    indices_path = os.path.join(MODEL_SAVE_PATH, 'feature_indices.pkl')
    
    if not os.path.exists(scaler_path) or not os.path.exists(indices_path):
        print("\n⚠️  Inference components not found. Running setup...")
        setup_inference()
    
    # Now load them
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Loaded scaler from {scaler_path}")
    
    with open(indices_path, 'rb') as f:
        feature_indices = pickle.load(f)
    print(f"Loaded {len(feature_indices)} feature indices from {indices_path}")
    
    return model, scaler, feature_indices


def extract_features_single(audio):
    """
    Extract 1D-LBP and 1D-LTP features from a single audio sample
    
    Args:
        audio: Audio signal array
    
    Returns:
        Feature vector
    """
    # Apply Butterworth bandpass filter
    filtered = butter_bandpass_filter(audio, BUTTER_LOW, BUTTER_HIGH, SAMPLE_RATE, BUTTER_ORDER)
    
    # Extract multi-scale 1D-LBP and 1D-LTP features
    features = extract_multi_scale_features(filtered)
    
    return features


def predict_audio_file(model, audio_path, scaler, feature_indices):
    """
    Make prediction for a single audio file
    
    Args:
        model: Trained model
        audio_path: Path to audio file
        scaler: StandardScaler for normalization
        feature_indices: Selected feature indices from ReliefF
    
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    # Load audio
    audio = load_audio_file(audio_path)
    
    if audio is None:
        raise ValueError(f"Could not load audio file: {audio_path}")
    
    # Extract features (full 2304 features)
    features = extract_features_single(audio)
    
    # Apply feature selection (select 256 features)
    features = features[feature_indices]
    
    # Normalize features using the trained scaler
    features = scaler.transform(features.reshape(1, -1))[0]
    
    # Reshape for model input (batch, features, 1)
    features = features.reshape(1, -1, 1)
    
    # Make prediction
    probabilities = model.predict(features, verbose=0)[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = CLASSES[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]
    
    return predicted_class, confidence, probabilities


def predict_batch(model, audio_paths, scaler, feature_indices):
    """
    Make predictions for multiple audio files
    
    Args:
        model: Trained model
        audio_paths: List of paths to audio files
        scaler: StandardScaler for normalization
        feature_indices: Selected feature indices
    
    Returns:
        List of prediction results
    """
    results = []
    
    for audio_path in audio_paths:
        try:
            pred_class, confidence, probs = predict_audio_file(
                model, audio_path, scaler, feature_indices
            )
            results.append({
                'file': audio_path,
                'prediction': pred_class,
                'confidence': confidence,
                'probabilities': dict(zip(CLASSES, probs.tolist()))
            })
        except Exception as e:
            results.append({
                'file': audio_path,
                'error': str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Heart Sound Classification Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py audio.wav
  python predict.py audio.wav --model-path models/best_model.keras
  python predict.py PASCAL/Atraining_normal/201101070538.wav
  python predict.py --setup  # Run once after training
        """
    )
    
    parser.add_argument(
        'audio_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to audio file (.wav)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model (default: models/best_model.keras)'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run setup to save scaler and feature indices (run once after training)'
    )
    
    args = parser.parse_args()
    
    # Handle setup
    if args.setup:
        setup_inference()
        return 0
    
    # Check if audio file provided
    if args.audio_file is None:
        parser.print_help()
        print("\n❌ Error: Please provide an audio file or use --setup")
        return 1
    
    # Check if file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1
    
    # Load model and preprocessing components
    try:
        model, scaler, feature_indices = load_model_and_components(args.model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Make prediction
    print(f"\n{'='*60}")
    print(f"Analyzing: {args.audio_file}")
    print(f"{'='*60}")
    
    try:
        pred_class, confidence, probs = predict_audio_file(
            model, args.audio_file, scaler, feature_indices
        )
        
        print(f"\n🔊 PREDICTION: {pred_class.upper()}")
        print(f"📊 Confidence: {confidence*100:.2f}%")
        print("\n📈 All class probabilities:")
        print("-" * 40)
        
        # Sort by probability
        class_probs = list(zip(CLASSES, probs))
        class_probs.sort(key=lambda x: x[1], reverse=True)
        
        for class_name, prob in class_probs:
            bar = "█" * int(prob * 30)
            marker = " ◄" if class_name == pred_class else ""
            print(f"  {class_name:12s}: {prob*100:6.2f}% {bar}{marker}")
        
        print(f"\n{'='*60}")
        
        # Clinical interpretation
        print("\n📋 Clinical Interpretation:")
        if pred_class == "normal":
            print("  ✅ Normal heart sounds detected (S1-S2)")
        elif pred_class == "murmur":
            print("  ⚠️  Heart murmur detected - Consider further cardiac evaluation")
        elif pred_class == "artifact":
            print("  ❌ Recording artifact detected - Consider re-recording")
        elif pred_class == "extrahls":
            print("  ⚠️  Extra heart sounds detected (S3/S4) - May indicate cardiac condition")
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
