"""
Configuration file for Heart Sound Classification
Based on: "Heart sounds classification using CNN with 1D-LBP and 1D-LTP features"
Er, Mehmet Bilal - Applied Acoustics 2021
"""
import os

# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset Configurations
DATASET_CONFIG = {
    'PASCAL': {
        'path': os.path.join(BASE_DIR, 'PASCAL'),
        'classes': ['normal', 'murmur', 'artifact', 'extrahls'],
        'class_dirs': {
            'normal': 'Atraining_normal',
            'murmur': 'Atraining_murmur',
            'artifact': 'Atraining_artifact',
            'extrahls': 'Atraining_extrahls'
        }
    },
    'Physionet': {
        'path': os.path.join(BASE_DIR, 'Physionet2016'),
        'classes': ['healthy', 'unhealthy'],
        'class_dirs': {
            'healthy': 'healthy',
            'unhealthy': 'unhealthy'
        },
        # Physionet has train/val split folders, handled in loader
        'subsets': ['train', 'val'] 
    }
}

# Default to PASCAL for backward compatibility if needed, 
# but code should now use get_dataset_config
DEFAULT_DATASET = 'PASCAL'
CLASSES = DATASET_CONFIG[DEFAULT_DATASET]['classes']
CLASS_DIRS = DATASET_CONFIG[DEFAULT_DATASET]['class_dirs']
DATA_DIR = DATASET_CONFIG[DEFAULT_DATASET]['path']
NUM_CLASSES = len(CLASSES)

# Audio parameters - from paper
SAMPLE_RATE = 2000  # Hz - PhysioNet uses 2kHz
DURATION = 5  # seconds

# Butterworth filter parameters (from paper Section 3.1)
BUTTER_ORDER = 5
BUTTER_LOW = 25  # Hz
BUTTER_HIGH = 400  # Hz

# 1D-LBP parameters (from paper Section 3.3)
LBP_NEIGHBORS = 8  # P = 8 neighbors

# 1D-LTP parameters (from paper Section 3.5)
LTP_THRESHOLD = 0.02  # threshold t for ternary pattern

# Feature selection - ReliefF
N_FEATURES_TO_SELECT = 256  # Number of top features to select

# CNN Training parameters (from paper Section 4.3)
BATCH_SIZE = 64  # Paper tested 32, 64, 128 - best was 64
LEARNING_RATE = 0.001  # Paper tested 0.1, 0.01, 0.001 - best was 0.001
EPOCHS = 140  # From paper
DROPOUT_RATE = 0.5  # 50% dropout as mentioned in paper

# Cross-validation
N_FOLDS = 10  # 10-fold cross validation as in paper
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model saving
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, 'best_model.keras')

# Random seed for reproducibility
RANDOM_SEED = 42
