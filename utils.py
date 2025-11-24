import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.neighbors import NearestNeighbors
import random

class Utils:
    """
    A utility class providing signal processing, feature extraction,
    and feature selection tools including filtering, LBP, LTP,
    and ReliefF feature-ranking.
    """
    # ---------------------------------------------------------
    #   BUTTERWORTH FILTER
    # ---------------------------------------------------------
    @staticmethod
    def butter_filter(signal, sr, lowcut=None, highcut=None, order=5):
        """
        Apply a Butterworth filter (lowpass, highpass, or bandpass) to a signal.

        Inputs:
            signal (np.ndarray): 1D raw signal.
            sr (int): Sampling rate of the signal (Hz).
            lowcut (float or None): Lower cutoff frequency in Hz.
            highcut (float or None): Upper cutoff frequency in Hz.
            order (int): Order of the Butterworth filter.

        Outputs:
            np.ndarray: Filtered signal (same length as input).
        """
        nyquist = 0.5 * sr
        
        # Determine filter type and normalized cutoff frequencies
        if lowcut is not None and highcut is not None:
            btype = 'band'
            Wn = [lowcut / nyquist, highcut / nyquist]
            # Validate bandpass frequencies
            if not (0 < Wn[0] < Wn[1] < 1):
                raise ValueError(f"Bandpass frequencies must be 0 < low < high < 1. Got: {Wn}")
        elif highcut is not None:
            btype = 'lowpass'
            Wn = highcut / nyquist
            # Validate lowpass frequency
            if not (0 < Wn < 1):
                raise ValueError(f"Lowpass cutoff frequency must be 0 < Wn < 1. Got: {Wn}")
        elif lowcut is not None:
            btype = 'highpass'
            Wn = lowcut / nyquist
            # Validate highpass frequency
            if not (0 < Wn < 1):
                raise ValueError(f"Highpass cutoff frequency must be 0 < Wn < 1. Got: {Wn}")
        else:
            raise ValueError("At least one of lowcut or highcut must be provided for filtering.")

        sos = butter(order, Wn, btype=btype, output='sos')
        return sosfiltfilt(sos, signal)

    # ---------------------------------------------------------
    #   TRUNCATE OR PAD
    # ---------------------------------------------------------
    @staticmethod
    def truncate_or_pad(signal, target_length):
        """
        Ensure signal has exactly `target_length` samples.

        Inputs:
            signal (np.ndarray): Input signal.
            target_length (int): Required output size.

        Outputs:
            np.ndarray: Truncated or zero-padded signal of length `target_length`.
        """
        if len(signal) >= target_length:
            return signal[:target_length]
        else:
            pad_len = target_length - len(signal)
            return np.pad(signal, (0, pad_len), mode='constant')

    # ---------------------------------------------------------
    #   LBP 1D
    # ---------------------------------------------------------
    @staticmethod
    def lbp_1d(signal, R):
        """
        Compute 1D Local Binary Pattern (LBP) for a signal.

        Inputs:
            signal (np.ndarray): Input 1D signal.
            R (int): Neighborhood radius.

        Outputs:
            np.ndarray (uint8): LBP values of length (N - 2R).
        """
        signal = np.asarray(signal)
        N = len(signal)

        M = N - 2 * R
        if M <= 0:
            return np.zeros(0, dtype=np.uint8)

        centers = signal[R : N - R]

        left_blocks = np.stack(
            [signal[R - k - 1 : N - R - k - 1] for k in range(R)],
            axis=1
        )

        right_blocks = np.stack(
            [signal[R + k + 1 : N - R + k + 1] for k in range(R)],
            axis=1
        )

        neighbors = np.concatenate([left_blocks, right_blocks], axis=1)
        bits = (neighbors >= centers[:, None]).astype(np.uint8)
        bits = bits[:, ::-1]

        weights = (1 << np.arange(2 * R))
        return bits @ weights

    # ---------------------------------------------------------
    #   LTP 1D
    # ---------------------------------------------------------
    @staticmethod
    def ltp_1d(signal, R, t):
        """
        Compute 1D Local Ternary Pattern (LTP) upper and lower patterns.

        Inputs:
            signal (np.ndarray): Input 1D signal.
            R (int): Neighborhood radius.
            t (float): Threshold for ternary comparison.

        Outputs:
            tuple[np.ndarray, np.ndarray]:
                upper_vals: LTP upper pattern values.
                lower_vals: LTP lower pattern values.
        """
        signal = np.asarray(signal)
        N = len(signal)

        M = N - 2 * R
        if M <= 0:
            return np.zeros((0,)), np.zeros((0,))

        centers = signal[R : N - R]

        left = np.stack([signal[R - k - 1 : N - R - k - 1] for k in range(R)], axis=1)
        right = np.stack([signal[R + k + 1 : N - R + k + 1] for k in range(R)], axis=1)

        neighbors = np.concatenate([left[:, ::-1], right], axis=1)

        lower_bound = centers[:, None] - t
        upper_bound = centers[:, None] + t

        upper_bits = (neighbors > upper_bound).astype(np.uint8)
        lower_bits = (neighbors < lower_bound).astype(np.uint8)

        weights = 2 ** np.arange(2 * R - 1, -1, -1)

        upper_vals = upper_bits @ weights
        lower_vals = lower_bits @ weights

        return upper_vals, lower_vals

    # ---------------------------------------------------------
    #   FULL FEATURE EXTRACTION PIPELINE
    # ---------------------------------------------------------
    @staticmethod
    def extract_features(signal, sr, R=4, threshold=0.5,
                             lowcut=25, highcut=400, order=5, duration_sec=9.0):
        """
        Full pipeline: signal → filtering → LBP + LTP → stacked → flattened.

        Inputs:
            signal (np.ndarray): Raw input audio signal.
            sr (int): Sampling rate.
            R (int): LBP/LTP radius.
            threshold (float): Ternary threshold for LTP.
            lowcut (float): High-pass cutoff frequency.
            highcut (float): Low-pass cutoff frequency.
            order (int): Filter order.
            duration_sec (float): Target duration of signal (seconds).

        Outputs:
            np.ndarray: Flattened feature vector of shape (3 * L,),
                        where L = target_length - 2R.
        """
        target_length = int(duration_sec * sr)
        signal = Utils.truncate_or_pad(signal, target_length)

        filtered = Utils.butter_filter(signal, sr,
                                        lowcut=lowcut,
                                        highcut=highcut,
                                        order=order)

        lbp = Utils.lbp_1d(filtered, R)
        ltp_up, ltp_down = Utils.ltp_1d(filtered, R, threshold)

        expected_L = max(0, target_length - 2 * R)

        lbp_ = Utils.truncate_or_pad(lbp, expected_L)
        ltp_up_ = Utils.truncate_or_pad(ltp_up, expected_L)
        ltp_down_ = Utils.truncate_or_pad(ltp_down, expected_L)

        X = np.stack([lbp_, ltp_up_, ltp_down_], axis=0)
        return X.flatten()

    # ---------------------------------------------------------
    #   RELIEFF
    # ---------------------------------------------------------
    @staticmethod
    def reliefF(X, y, n_neighbors=10, n_iterations=100, seed=None):
        """
        ReliefF feature selection algorithm.

        Inputs:
            X (np.ndarray): Feature matrix of shape (N_samples, N_features).
            y (np.ndarray): Class labels of shape (N_samples,).
            n_neighbors (int): Number of neighbors to consider.
            n_iterations (int): Random sample iterations for scoring.
            seed (int or None): Random seed.

        Outputs:
            np.ndarray: Feature importance weights (length N_features).
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        weights = np.zeros(n_features)

        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(X)

        for _ in range(n_iterations):
            idx = random.randint(0, n_samples - 1)
            R_i = X[idx]
            class_R = y[idx]

            distances, indices = nn.kneighbors(R_i.reshape(1, -1))
            neighbor_idx = indices[0][1:]
            neighbor_classes = y[neighbor_idx]

            hits_idx = neighbor_idx[neighbor_classes == class_R][:n_neighbors]
            hits = X[hits_idx]

            misses_by_class = {}
            for cls in unique_classes:
                if cls != class_R:
                    miss_idx = neighbor_idx[neighbor_classes == cls][:n_neighbors]
                    misses_by_class[cls] = X[miss_idx]

            if len(hits) == 0 or any(len(v) == 0 for v in misses_by_class.values()):
                continue

            for f_idx in range(n_features):
                diff_hit = np.mean(np.abs(R_i[f_idx] - hits[:, f_idx])) if len(hits) > 0 else 0

                diff_miss_sum = 0
                for cls in unique_classes:
                    if cls == class_R:
                        continue
                    P = (np.sum(y == cls) / n_samples)
                    P_ref = (np.sum(y == class_R) / n_samples)
                    P_factor = P / (1 - P_ref)

                    misses = misses_by_class[cls]
                    diff_miss = np.mean(np.abs(R_i[f_idx] - misses[:, f_idx]))
                    diff_miss_sum += P_factor * diff_miss

                weights[f_idx] += diff_miss_sum - diff_hit

        weights /= n_iterations
        weights[weights < 0] = 0
        return weights

    # ---------------------------------------------------------
    #   SELECT TOP RELIEFF FEATURES
    # ---------------------------------------------------------
    @staticmethod
    def select_features_reliefF(X, y, n_selected_features,
                                 n_neighbors=10, n_iterations=100, seed=None):
        """
        Select top-N features using ReliefF ranking.

        Inputs:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
            n_selected_features (int): Number of top features to select.
            n_neighbors (int): Neighbors for ReliefF.
            n_iterations (int): Iterations for ReliefF.
            seed (int or None): Random seed.

        Outputs:
            tuple:
                selected_indices (np.ndarray): Indices of selected features.
                weights (np.ndarray): ReliefF feature weights.
        """
        if n_selected_features <= 0:
            return np.array([]), np.array([])

        if n_selected_features > X.shape[1]:
            n_selected_features = X.shape[1]

        weights = Utils.reliefF(
            X, y,
            n_neighbors=n_neighbors,
            n_iterations=n_iterations,
            seed=seed
        )

        sorted_idx = np.argsort(weights)[::-1]
        selected = sorted_idx[:n_selected_features]

        return selected, weights