import os
import sys
import datetime
import numpy as np
import librosa
import glob
from sklearn.preprocessing import StandardScaler

# Import your utility class
from utils import Utils

# ============================================================
#               ENABLE TERMINAL PRINT LOGGING
# ============================================================

# Ensure the 'outputs' directory exists for logs and results
os.makedirs('outputs', exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"outputs/run_log_{timestamp}.txt"

class Logger(object):
    """
    Logger that writes prints to both console and a log file.
    """
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass  # no special flush behavior needed

# Redirect all prints to our logger
sys.stdout = Logger(LOG_FILE)


# ============================================================
#                      MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    print("\n======================================================")
    print("   Heart Sound Classification - Feature Extraction")
    print("======================================================\n")
    
    print("Select Dataset:")
    print("  [1] PhysioNet2016")
    print("  [2] PASCAL")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        DATASET_NAME = "PhysioNet2016"
        dataset_root = "PhysioNet2016"
        # PhysioNet specific config
        BASE_DIR = os.path.join(dataset_root, "train")
        DATA_SUBDIRS = ["healthy", "unhealthy"]
        LABEL_MAP = {"healthy": 0, "unhealthy": 1}
        SR = 2000
        DURATION_SEC = 5.0
        AUGMENT_DATA = True
        OUTPUT_PREFIX = "PhysioNet"
        
    elif choice == "2":
        DATASET_NAME = "PASCAL"
        dataset_root = "PASCAL"
        # PASCAL specific config
        BASE_DIR = dataset_root
        DATA_SUBDIRS = ["Atraining_normal", "Atraining_murmur", "Atraining_extrahls", "Atraining_artifact"]
        LABEL_MAP = {
            "Atraining_normal": 0,
            "Atraining_murmur": 1,
            "Atraining_extrahls": 2,
            "Atraining_artifact": 3,
        }
        SR = 4000
        DURATION_SEC = 9.0
        AUGMENT_DATA = False
        OUTPUT_PREFIX = "PASCAL"
        
    else:
        print("[ERROR] Invalid choice. Exiting.")
        sys.exit(1)

    # Common Parameters
    LOWCUT = 25
    HIGHCUT = 400
    FILTER_ORDER = 5
    R_NEIGHBORS = 4
    THRESHOLD = 0.5
    
    # ReliefF Parameters
    RELIEF_N_NEIGHBORS = 10
    RELIEF_N_ITERATIONS = 500 if DATASET_NAME == "PASCAL" else 100 # Lower iterations for larger dataset
    N_SELECTED_FEATURES = 100
    RELIEF_SEED = 42

    # Derived
    target_signal_length = int(SR * DURATION_SEC)
    if target_signal_length - 2 * R_NEIGHBORS > 0:
        NUM_FEATURES_PER_SAMPLE = int((target_signal_length - 2 * R_NEIGHBORS) * 3)
    else:
        NUM_FEATURES_PER_SAMPLE = 0

    inv_label_map = {v: k.replace("Atraining_", "") for k, v in LABEL_MAP.items()}

    print(f"\n=========== Processing {DATASET_NAME} Dataset ===========\n")
    print(f"Sampling Rate: {SR}, Duration: {DURATION_SEC}s")
    print(f"LBP/LTP R={R_NEIGHBORS}, Threshold={THRESHOLD}")
    print(f"Filter: {LOWCUT}-{HIGHCUT} Hz, Order={FILTER_ORDER}")
    print(f"Augmentation: {AUGMENT_DATA}")
    print(f"Expected features per sample: {NUM_FEATURES_PER_SAMPLE}")
    print(f"Log saved to: {LOG_FILE}")
    print("------------------------------------------------------\n")

    all_features = []
    all_labels = []
    
    processed_count = 0
    skipped_count = 0

    if not os.path.isdir(BASE_DIR):
        print(f"[ERROR] Cannot find base directory: {BASE_DIR}")
        sys.exit(1)

    # -------------------------------------------------------
    #   Scan folders + Extract Features
    # -------------------------------------------------------
    for sub_dir in DATA_SUBDIRS:
        dir_path = os.path.join(BASE_DIR, sub_dir)
        # For PASCAL, sub_dir is directly in root. For PhysioNet, it's in 'train/'.
        # The logic above handles BASE_DIR correctly for both.
        
        if not os.path.isdir(dir_path):
             # Try checking if it's just a folder name in the root (for PASCAL mainly if structure varies)
             # But BASE_DIR should be correct.
             print(f"  [WARN] Folder not found: {dir_path}")
             continue

        current_label = LABEL_MAP[sub_dir]
        print(f"\nProcessing Folder: {sub_dir} (Label {current_label})")

        files = glob.glob(os.path.join(dir_path, "*.wav"))
        
        for wav_file in files:
            try:
                # 1. Load Audio
                audio, current_sr = librosa.load(wav_file, sr=SR, duration=DURATION_SEC) # Load with duration limit
                
                # 2. Extract Features (LBP + LTP)
                features = Utils.extract_features(
                    audio, current_sr,
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
                all_labels.append(current_label)
                
                # 4. Data Augmentation (if enabled)
                if AUGMENT_DATA:
                    aug_audio = Utils.augment_audio(audio, SR)
                    features_aug = Utils.extract_features(
                        aug_audio, SR,
                        R=R_NEIGHBORS,
                        threshold=THRESHOLD,
                        lowcut=LOWCUT,
                        highcut=HIGHCUT,
                        order=FILTER_ORDER,
                        duration_sec=DURATION_SEC
                    )
                    if len(features_aug) != NUM_FEATURES_PER_SAMPLE:
                        features_aug = Utils.truncate_or_pad(features_aug, NUM_FEATURES_PER_SAMPLE)
                        
                    all_features.append(features_aug)
                    all_labels.append(current_label)

                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} files...")

            except Exception as e:
                print(f"ERROR processing {wav_file}: {e}")
                skipped_count += 1

    # Convert to arrays
    X_dataset = np.array(all_features)
    y_dataset = np.array(all_labels)

    print("\n========== Dataset Summary ==========")
    print(f"Processed: {processed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"X shape:   {X_dataset.shape}")
    print(f"y shape:   {y_dataset.shape}")

    if len(all_labels) == 0:
        print("[ERROR] No data processed. Exiting.")
        sys.exit(1)

    print("\nClass Distribution:")
    unique_labels, counts = np.unique(y_dataset, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label} ({inv_label_map.get(label, 'Unknown')}): {count}")

    # --- Feature Selection (ReliefF) ---
    print(f"\nRunning ReliefF Feature Selection (selecting top {N_SELECTED_FEATURES})...")
    
    # Standardize first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dataset)
    
    selected_indices, weights = Utils.select_features_reliefF(
        X_scaled, y_dataset, 
        n_selected_features=N_SELECTED_FEATURES, 
        n_neighbors=RELIEF_N_NEIGHBORS, 
        n_iterations=RELIEF_N_ITERATIONS,
        seed=RELIEF_SEED
    )
    
    X_reduced = X_scaled[:, selected_indices]
    print(f"Features reduced from {X_dataset.shape[1]} to {X_reduced.shape[1]}")

    # --- Save Outputs ---
    print("\nSaving outputs...")
    np.savetxt(f"outputs/{OUTPUT_PREFIX}_X_dataset.csv", X_dataset, delimiter=",")
    np.savetxt(f"outputs/{OUTPUT_PREFIX}_y.csv", y_dataset, delimiter=",", fmt="%d") # Unified name _y.csv
    np.savetxt(f"outputs/{OUTPUT_PREFIX}_feature_weights.csv", weights, delimiter=",")
    np.savetxt(f"outputs/{OUTPUT_PREFIX}_selected_feature_indices.csv", selected_indices, delimiter=",", fmt="%d")
    np.savetxt(f"outputs/{OUTPUT_PREFIX}_X_reduced.csv", X_reduced, delimiter=",")
    
    print(f"Saved: {OUTPUT_PREFIX}_X_dataset.csv, {OUTPUT_PREFIX}_y.csv, {OUTPUT_PREFIX}_feature_weights.csv, {OUTPUT_PREFIX}_selected_feature_indices.csv, {OUTPUT_PREFIX}_X_reduced.csv")

    print("\n=========== PROCESS COMPLETE ===========")
    print("Your data is ready for CNN training.")