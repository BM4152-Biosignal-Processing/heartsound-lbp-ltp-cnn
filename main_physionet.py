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
#                GLOBAL CONFIGURATION
# ============================================================

# Select Dataset: "PhysioNet2016" or "PASCAL"
DATASET_NAME = "PhysioNet2016" 
# DATASET_NAME = "PASCAL"

if DATASET_NAME == "PhysioNet2016":
    dataset_root = "PhysioNet2016"
    # We look into 'train' folder. 'val' can be used for validation if needed.
    # Structure: PhysioNet2016/train/healthy, PhysioNet2016/train/unhealthy
    BASE_DIR = os.path.join(dataset_root, "train")
    DATA_SUBDIRS = ["healthy", "unhealthy"]
    LABEL_MAP = {"healthy": 0, "unhealthy": 1}
    CLASS_NAMES = ["Healthy", "Unhealthy"]
    
elif DATASET_NAME == "PASCAL":
    dataset_root = "PASCAL"
    BASE_DIR = dataset_root
    # PASCAL B Dataset structure usually
    DATA_SUBDIRS = ["Atraining_normal", "Atraining_murmur", "Atraining_extrahls", "Atraining_artifact"]
    # Mapping for 2-Class (Normal vs Abnormal) or Multi-class
    # Paper often does Normal vs Abnormal. Let's do Multi-class first or map to 2.
    # If 2-class: Normal=0, Murmur/Extra=1. Artifacts usually removed.
    # Let's stick to the folder structure labels for now.
    LABEL_MAP = {
        "Atraining_normal": 0,
        "Atraining_murmur": 1,
        "Atraining_extrahls": 2,
        "Atraining_artifact": 3
    }
    CLASS_NAMES = ["Normal", "Murmur", "ExtraHLS", "Artifact"]

# ---- Audio + Feature Extraction Parameters ---- 
SR = 2000 # Downsample to 2000Hz as per many papers
DURATION_SEC = 5.0 # Fixed duration
LOWCUT = 25
HIGHCUT = 400
FILTER_ORDER = 5
AUGMENT_DATA = True # Enable augmentation for training data

# ---- Derived Feature Dimensions ----
target_signal_length = int(SR * DURATION_SEC)

# ============================================================
#                      MAIN SCRIPT EXECUTION
# ============================================================

if __name__ == "__main__":

    print(f"\n=========== Processing Dataset: {DATASET_NAME} ===========\n")

    X_spectrograms = []
    X_mfccs = []
    y_labels = []
    
    processed_count = 0
    skipped_count = 0

    if not os.path.isdir(BASE_DIR):
        print(f"❌ ERROR: Cannot find base directory: {BASE_DIR}")
        sys.exit(1)

    print(f"Scanning directories in {BASE_DIR}: {DATA_SUBDIRS}")

    for sub_dir in DATA_SUBDIRS:
        dir_path = os.path.join(BASE_DIR, sub_dir)
        if not os.path.isdir(dir_path):
            print(f"  ⚠️ Warning: Folder not found: {dir_path}")
            continue
            
        current_label = LABEL_MAP[sub_dir]
        print(f"Processing folder: {sub_dir} (Label: {current_label})")
        
        files = glob.glob(os.path.join(dir_path, "*.wav"))
        
        for wav_file in files:
            try:
                # 1. Load Audio
                audio, _ = librosa.load(wav_file, sr=SR, duration=DURATION_SEC)
                
                # 2. Preprocessing
                audio = Utils.butter_filter(audio, SR, LOWCUT, HIGHCUT, order=FILTER_ORDER)
                audio = Utils.truncate_or_pad(audio, target_signal_length)
                
                # 3. Feature Extraction (Original)
                spec = Utils.get_spectrogram(audio, sr=SR)
                mfcc = Utils.get_mfcc(audio, sr=SR)
                
                X_spectrograms.append(spec)
                X_mfccs.append(mfcc)
                y_labels.append(current_label)
                
                # 4. Data Augmentation (Only for training, but here we process all)
                # To reach >96%, augmentation is crucial.
                if AUGMENT_DATA:
                    # Create 1 augmented version per file
                    aug_audio = Utils.augment_audio(audio, SR)
                    spec_aug = Utils.get_spectrogram(aug_audio, sr=SR)
                    mfcc_aug = Utils.get_mfcc(aug_audio, sr=SR)
                    
                    X_spectrograms.append(spec_aug)
                    X_mfccs.append(mfcc_aug)
                    y_labels.append(current_label)

                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} files...")
                    
            except Exception as e:
                print(f"  Error processing {wav_file}: {e}")
                skipped_count += 1

    print(f"\nDone. Processed files: {processed_count}, Total samples (with aug): {len(y_labels)}")

    # -------------------------------------------------------
    #   Save Data for Training
    # -------------------------------------------------------
    if len(y_labels) > 0:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save with dataset name prefix to distinguish
        prefix = DATASET_NAME
        
        print(f"Saving features to '{output_dir}/' with prefix '{prefix}'...")
        
        X_spec_arr = np.array(X_spectrograms)
        X_mfcc_arr = np.array(X_mfccs)
        y_arr = np.array(y_labels)
        
        np.save(f"{output_dir}/{prefix}_X_spectrogram.npy", X_spec_arr)
        np.save(f"{output_dir}/{prefix}_X_mfcc.npy", X_mfcc_arr)
        np.save(f"{output_dir}/{prefix}_y_dataset.npy", y_arr)
        
        print(f"Saved {prefix}_X_spectrogram.npy: {X_spec_arr.shape}")
        print(f"Saved {prefix}_X_mfcc.npy: {X_mfcc_arr.shape}")
        print(f"Saved {prefix}_y_dataset.npy: {y_arr.shape}")
    else:
        print("No data processed.")

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