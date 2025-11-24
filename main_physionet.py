import os
import sys
import datetime
import numpy as np
import librosa
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# You must ensure the 'Utils' class is available from your utils.py file
from utils import Utils 


# ============================================================
#               ENABLE TERMINAL PRINT LOGGING
# ============================================================

os.makedirs('outputs', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"outputs/run_log_PN2016_2CLASS_Folder_{timestamp}.txt"

# Simple class to log output to both console and a file
class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
sys.stdout = Logger(LOG_FILE)


# ============================================================
#                GLOBAL CONFIGURATION (2-Class)
# ============================================================

# ---- Dataset Paths ----
dataset_root = "PhysioNet2016"        
DATA_SUBDIRS = ["healthy", "unhealthy"] # Folders to scan

# ---- Class Label Mapping (2 Classes) ----
N_CLASSES = 2
# Labels are assigned based on the folder name
LABEL_MAP = {
    "healthy": 0,    # Normal
    "unhealthy": 1,  # Abnormal
}

# Friendly names for printing
inv_label_map = {0: "Healthy (Normal)", 1: "Unhealthy (Abnormal)"}

# ---- Audio + Feature Extraction Parameters ---- 
SR = 4000
DURATION_SEC = 9.0
R_NEIGHBORS = 4
THRESHOLD = 0.5
LOWCUT = 25
HIGHCUT = 400
FILTER_ORDER = 5

# ---- Derived Feature Dimensions ----
target_signal_length = int(SR * DURATION_SEC)
expected_L = max(0, target_signal_length - 2 * R_NEIGHBORS) 
NUM_FEATURES_PER_SAMPLE = 3 * expected_L

# ---- ReliefF Parameters ----
RELIEF_N_NEIGHBORS = 10
RELIEF_N_ITERATIONS = 500
N_SELECTED_FEATURES = 100 # Target feature count for CNN
RELIEF_SEED = 42


# ============================================================
#                      MAIN SCRIPT EXECUTION
# ============================================================

if __name__ == "__main__":

    print(f"\n=========== PhysioNet 2016 Dataset Processing (2-Class, Folder-Based) ===========\n")

    all_features = []
    all_labels = []
    processed_count = 0
    skipped_count = 0
    
    # Base directory for audio files
    train_dir = os.path.join(dataset_root, "train")

    if not os.path.isdir(train_dir):
        print(f"❌ ERROR: Cannot find training directory: {train_dir}")
        sys.exit(1)

    print(f"Scanning directories: {DATA_SUBDIRS}")

    # -------------------------------------------------------
    #   Scan folders, Assign Labels, and Extract Features
    # -------------------------------------------------------

    for folder_name in DATA_SUBDIRS:
        
        current_label = LABEL_MAP.get(folder_name)
        if current_label is None:
            print(f"Warning: Skipping unknown folder {folder_name}")
            continue
            
        folder_path = os.path.join(train_dir, folder_name)
        
        # Find all .wav files in the current folder
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        
        print(f"\nProcessing {len(wav_files)} files in '{folder_name}' (Label: {current_label})...")

        for filepath in wav_files:
            
            try:
                # 1. Load Audio
                signal, current_sr = librosa.load(filepath, sr=SR, mono=True)

                # 2. Extract Features (1D-LBP/1D-LTP)
                features = Utils.extract_features(
                    signal, current_sr,
                    R=R_NEIGHBORS,
                    threshold=THRESHOLD,
                    lowcut=LOWCUT,
                    highcut=HIGHCUT,
                    order=FILTER_ORDER,
                    duration_sec=DURATION_SEC
                )

                # 3. Guarantee consistent size
                if len(features) != NUM_FEATURES_PER_SAMPLE:
                    features = Utils.truncate_or_pad(features, NUM_FEATURES_PER_SAMPLE)

                all_features.append(features)
                all_labels.append(current_label) # Label assigned based on folder
                processed_count += 1

            except Exception as e:
                print(f"ERROR processing {filepath}: {e}")
                skipped_count += 1
                
        if processed_count > 0 and processed_count % 100 == 0:
             print(f"Progress: Processed {processed_count} files.")


    X_dataset = np.array(all_features)
    y_dataset = np.array(all_labels)

    # -------------------------------------------------------
    #   Summary, Standardization, and ReliefF
    # -------------------------------------------------------
    
    print("\n========== Dataset Summary ==========")
    print(f"Processed files: {processed_count}")
    print(f"Skipped files: {skipped_count}")
    if processed_count == 0: sys.exit(1)
    
    # ... (rest of summary printing and ReliefF remains the same) ...
    
    print("\nClass Distribution:")
    unique_labels, counts = np.unique(y_dataset, return_counts=True)
    for lbl, count in zip(unique_labels, counts):
        print(f"  {inv_label_map.get(lbl, f'Unknown({lbl})')}: {count}")
    print("--------------------------------------\n")


    # ============================================================
    #                 STANDARDIZATION & RELIEFF
    # ============================================================
    print("--- Standardizing Features ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dataset)

    print("\n--- Running ReliefF Feature Selection ---")
    selected_indices, feature_weights = Utils.select_features_reliefF(
        X_scaled, y_dataset, n_selected_features=N_SELECTED_FEATURES, 
        n_neighbors=RELIEF_N_NEIGHBORS, n_iterations=RELIEF_N_ITERATIONS, seed=RELIEF_SEED
    )

    X_reduced = X_scaled[:, selected_indices]
    print(f"✅ ReliefF Selected: {len(selected_indices)} features. Reduced dataset shape: {X_reduced.shape}")


    # ============================================================
    #                 SAVE FINAL OUTPUTS
    # ============================================================
    # Use 'folder' in the filename to distinguish this run
    OUTPUT_X = "outputs/X_reduced_PN2016_2class_folder.csv"
    OUTPUT_Y = "outputs/y_dataset_PN2016_2class_folder.csv"
    
    np.savetxt(OUTPUT_X, X_reduced, delimiter=",")
    np.savetxt(OUTPUT_Y, y_dataset, delimiter=",")
    print(f"Saved {OUTPUT_X} and {OUTPUT_Y}")

    print("\n=========== Feature Processing COMPLETE ===========")