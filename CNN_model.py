import numpy as np
import pandas as pd
from utils import Utils


# ================================================================
# SETTINGS
# ================================================================
TOP_K = 500   # Number of selected features for CNN
RELIEFF_CSV = "ReliefF_scores_mode_1.csv"  # or mode_2
DATASET_MODE = 1   # 1 = PASCAL, 2 = PhysioNet


# ================================================================
# LOAD RELIEFF SCORES
# ================================================================
def load_relief_scores(csv_path):
    df = pd.read_csv(csv_path)
    scores = df["score"].values
    return scores


# ================================================================
# LOAD X AND y AGAIN (REQUIRED TO SELECT TOP FEATURES)
# ================================================================
def extract_features(signal, sr=4000, target_len=36000):
    signal = Utils.butter_filter(signal, sr, 25, 400)
    signal = Utils.truncate_or_pad(signal, target_len)
    lbp = Utils.lbp_1d(signal)
    up, low = Utils.ltp_1d(signal)
    return np.concatenate([lbp, up, low])


def load_dataset(mode):
    import librosa, os
    X = []
    y = []

    SR = 4000
    TARGET_LEN = 9 * SR

    if mode == 1:
        base = "PASCAL"
        folder_map = {
            "Atraining_artifact": 0,
            "Atraining_extrahls": 1,
            "Atraining_murmur": 2,
            "Atraining_normal": 3
        }
    else:
        base = "Physionet"
        folder_map = {
            "train/healthy": 0,
            "train/unhealthy": 1,
            "val/healthy": 0,
            "val/unhealthy": 1
        }

    for folder, label in folder_map.items():
        fpath = os.path.join(base, folder)
        for file in os.listdir(fpath):
            if not file.endswith(".wav"):
                continue

            sig, _ = librosa.load(os.path.join(fpath, file), sr=SR)
            feat = extract_features(sig)
            X.append(feat)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ================================================================
# CREATE CNN INPUT
# ================================================================
def create_cnn_input(X, y, scores, top_k):
    # Sort feature importance (descending)
    idx = np.argsort(scores)[::-1]
    selected = idx[:top_k]

    print(f"Selected top {top_k} features.")

    # Reduce X → only top-K features
    X_reduced = X[:, selected]

    # CNN expects (samples, length, channels)
    X_cnn = X_reduced[..., None]  # Add channel dim

    return X_cnn, y


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":

    print("Loading ReliefF scores...")
    scores = load_relief_scores(RELIEFF_CSV)

    print("Loading full dataset again...")
    X, y = load_dataset(DATASET_MODE)
    print("Dataset:", X.shape, y.shape)

    print("Creating CNN input tensor...")
    X_cnn, y_cnn = create_cnn_input(X, y, scores, TOP_K)

    print("Final CNN input shape:", X_cnn.shape)
    print("Labels shape:", y_cnn.shape)

    # Optional: Save the ready-to-train arrays
    np.save("X_cnn.npy", X_cnn)
    np.save("y_cnn.npy", y_cnn)

    print("Saved X_cnn.npy and y_cnn.npy.")
