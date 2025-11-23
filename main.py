import os
import sys
import csv
import datetime
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from utils import Utils   # Your utility class


# ============================================================
#                ENABLE TERMINAL PRINT LOGGING
# ============================================================

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
#                 GLOBAL CONFIGURATION
# ============================================================

# ---- Dataset Paths ----
dataset_root = "PASCAL"        # Root folder of dataset

# ---- Class Label Mapping ----
label_map = {
    "Atraining_normal": 0,
    "Atraining_murmur": 1,
    "Atraining_extrahls": 2,
    "Atraining_artifact": 3,
}

# Friendly names for printing
inv_label_map = {v: k.replace("Atraining_", "") for k, v in label_map.items()}

# ---- Audio + Feature Extraction Parameters ----
SR = 4000                    # Sampling rate
DURATION_SEC = 9.0           # Standard length for all audio
R_NEIGHBORS = 4              # LBP/LTP radius
THRESHOLD = 0.5              # LTP threshold
LOWCUT = 25                  # High-pass band
HIGHCUT = 400                # Low-pass band
FILTER_ORDER = 5             # Butterworth order

# ---- Derived Feature Dimensions ----
target_signal_length = int(SR * DURATION_SEC)
expected_L = max(0, target_signal_length - 2 * R_NEIGHBORS)
NUM_FEATURES_PER_SAMPLE = 3 * expected_L

# ---- ReliefF Parameters ----
RELIEF_N_NEIGHBORS = 10
RELIEF_N_ITERATIONS = 500
N_SELECTED_FEATURES = 100
RELIEF_SEED = 42



# ============================================================
#                        MAIN SCRIPT
# ============================================================

if __name__ == "__main__":

    print(f"\n=========== PASCAL Dataset Processing ===========\n")
    print(f"Sampling Rate: {SR}, Duration: {DURATION_SEC}s")
    print(f"LBP/LTP R={R_NEIGHBORS}, Threshold={THRESHOLD}")
    print(f"Filter: {LOWCUT}-{HIGHCUT} Hz, Order={FILTER_ORDER}")
    print(f"Expected features per sample: {NUM_FEATURES_PER_SAMPLE}")
    print(f"Log saved to: {LOG_FILE}")
    print("------------------------------------------------------\n")

    all_features = []
    all_labels = []
    all_filepaths = []

    processed_count = 0
    skipped_count = 0

    # -------------------------------------------------------
    #     Scan folders + Extract Features
    # -------------------------------------------------------
    for folder_name in sorted(os.listdir(dataset_root)):
        folder_path = os.path.join(dataset_root, folder_name)

        if os.path.isdir(folder_path) and folder_name in label_map:
            label = label_map[folder_name]
            print(f"\nProcessing Folder: {folder_name} (Label {label})")

            for filename in os.listdir(folder_path):
                if filename.endswith(".wav"):
                    filepath = os.path.join(folder_path, filename)
                    all_filepaths.append(filepath)

                    try:
                        # Load audio file
                        signal, current_sr = librosa.load(filepath, sr=SR, mono=True)

                        # Extract features
                        features = Utils.extract_features(
                            signal, current_sr,
                            R=R_NEIGHBORS,
                            threshold=THRESHOLD,
                            lowcut=LOWCUT,
                            highcut=HIGHCUT,
                            order=FILTER_ORDER,
                            duration_sec=DURATION_SEC
                        )

                        # Guarantee consistent size
                        if len(features) != NUM_FEATURES_PER_SAMPLE:
                            print(f"Fixing feature size mismatch for: {filename}")
                            features = Utils.truncate_or_pad(features, NUM_FEATURES_PER_SAMPLE)

                        all_features.append(features)
                        all_labels.append(label)
                        processed_count += 1

                    except Exception as e:
                        print(f"ERROR processing {filepath}: {e}")
                        skipped_count += 1
        else:
            if os.path.isdir(folder_path):
                print(f"Skipping: {folder_name} (Not a valid class)")

    # Convert to arrays
    X_dataset = np.array(all_features)
    y_dataset = np.array(all_labels)

    print("\n========== Dataset Summary ==========")
    print(f"Processed: {processed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"X shape:   {X_dataset.shape}")
    print(f"y shape:   {y_dataset.shape}")

    print("\nClass Distribution:")
    unique_labels, counts = np.unique(y_dataset, return_counts=True)
    for lbl, count in zip(unique_labels, counts):
        print(f"  {inv_label_map[lbl]}: {count}")

    print("--------------------------------------\n")



    # ============================================================
    #                SAVE RAW FEATURES AS CSV
    # ============================================================
    np.savetxt("outputs/X_dataset.csv", X_dataset, delimiter=",")
    np.savetxt("outputs/y_dataset.csv", y_dataset, delimiter=",")
    print("Saved X_dataset.csv + y_dataset.csv to outputs/")



    # ============================================================
    #                STANDARDIZE THE FEATURES
    # ============================================================
    print("\n--- Standardizing Features ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dataset)
    print("Feature scaling complete.")



    # ============================================================
    #                APPLY RELIEFF
    # ============================================================
    print("\n--- Running ReliefF Feature Selection ---")
    print(f"Using {RELIEF_N_NEIGHBORS} neighbors, {RELIEF_N_ITERATIONS} iterations")
    print(f"Selecting top {N_SELECTED_FEATURES} features\n")

    selected_indices, feature_weights = Utils.select_features_reliefF(
        X_scaled,
        y_dataset,
        n_selected_features=N_SELECTED_FEATURES,
        n_neighbors=RELIEF_N_NEIGHBORS,
        n_iterations=RELIEF_N_ITERATIONS,
        seed=RELIEF_SEED
    )

    print(f"ReliefF Selected: {len(selected_indices)} features")


    # ============================================================
    #                SAVE RELIEFF outputs
    # ============================================================
    np.savetxt("outputs/feature_weights.csv", feature_weights, delimiter=",")
    np.savetxt("outputs/selected_feature_indices.csv", selected_indices, delimiter=",")
    print("Saved ReliefF results to outputs/")



    # ============================================================
    #                PLOT FEATURE WEIGHTS
    # ============================================================
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(feature_weights)), feature_weights)
    plt.title("ReliefF Feature Weights")
    plt.xlabel("Feature Index")
    plt.ylabel("Weight")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()



    # ============================================================
    #                REDUCE DATASET
    # ============================================================
    X_reduced = X_scaled[:, selected_indices]
    print(f"Reduced dataset shape: {X_reduced.shape}")


    # Save reduced dataset
    np.savetxt("outputs/X_reduced.csv", X_reduced, delimiter=",")
    print("Saved X_reduced.csv to outputs/")


    print("\n=========== PROCESS COMPLETE ===========")
    print("Your data is ready for CNN training.")
