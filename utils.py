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

        # Neighbors: P = 2*R
        # We collect R neighbors to the left and R neighbors to the right
        # left_blocks shape: (M, R)
        left_blocks = np.stack(
            [signal[R - k - 1 : N - R - k - 1] for k in range(R)],
            axis=1
        )

        # right_blocks shape: (M, R)
        right_blocks = np.stack(
            [signal[R + k + 1 : N - R + k + 1] for k in range(R)],
            axis=1
        )

        # Concatenate to get all 2R neighbors
        # Order: [left_R, ..., left_1, right_1, ..., right_R]
        neighbors = np.concatenate([left_blocks, right_blocks], axis=1)
        
        # Compare neighbors with center
        bits = (neighbors >= centers[:, None]).astype(np.uint8)
        
        # Convert bits to decimal
        # We want the first neighbor (left-most) to be the MSB or LSB?
        # Standard LBP usually assigns weights 2^0, 2^1, ...
        # Let's assign 2^0 to the first column, 2^1 to the second, etc.
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

        neighbors = np.concatenate([left, right], axis=1)

        lower_bound = centers[:, None] - t
        upper_bound = centers[:, None] + t

        # Ternary coding:
        # neighbor > center + t  --> +1
        # neighbor < center - t  --> -1
        # else                   -->  0
        
        # Split into Upper and Lower binary patterns
        # Upper: 1 if neighbor > center + t, else 0
        upper_bits = (neighbors > upper_bound).astype(np.uint8)
        
        # Lower: 1 if neighbor < center - t, else 0
        lower_bits = (neighbors < lower_bound).astype(np.uint8)

        weights = (1 << np.arange(2 * R))

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

        # Ensure all are same length (should be if logic is correct)
        lbp_ = Utils.truncate_or_pad(lbp, expected_L)
        ltp_up_ = Utils.truncate_or_pad(ltp_up, expected_L)
        ltp_down_ = Utils.truncate_or_pad(ltp_down, expected_L)

        # Stack: [LBP, LTP_Upper, LTP_Lower]
        # Shape: (3, L)
        X = np.stack([lbp_, ltp_up_, ltp_down_], axis=0)
        
        # Flatten: (3*L,)
        # We want to preserve the temporal structure for 1D CNN?
        # The paper says "1D CNN with 1D-LBP and 1D-LTP features".
        # Usually this means the input to CNN is (L, 3) or (3, L).
        # But for ReliefF (feature selection), we need a 1D vector per sample.
        # So flattening is correct for Feature Selection.
        # Later for CNN, we might reshape it back or use 1D CNN on the flattened vector 
        # (treating it as a long sequence) or reshape to (L, 3).
        # Given the "ReliefF" step, it selects "top features". 
        # If we select random features from the flattened vector, we lose temporal structure.
        # However, the prompt says "Relief feature selection then a 1D CNN".
        # This implies we select specific *points* in the LBP/LTP streams that are most discriminative.
        
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

        # Use all samples for NN search
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(X)

        for _ in range(n_iterations):
            idx = random.randint(0, n_samples - 1)
            R_i = X[idx]
            class_R = y[idx]

            distances, indices = nn.kneighbors(R_i.reshape(1, -1))
            # indices[0][0] is the point itself
            neighbor_idx = indices[0][1:]
            neighbor_classes = y[neighbor_idx]

            hits_idx = neighbor_idx[neighbor_classes == class_R][:n_neighbors]
            hits = X[hits_idx]

            misses_by_class = {}
            for cls in unique_classes:
                if cls != class_R:
                    miss_idx = neighbor_idx[neighbor_classes == cls][:n_neighbors]
                    misses_by_class[cls] = X[miss_idx]

            # Skip if not enough neighbors
            if len(hits) == 0:
                continue

            # Update weights
            # W[A] = W[A] - diff(A, R, H) + sum(P(C)*diff(A, R, M(C)))
            
            # Diff Hit
            diff_hit = np.mean(np.abs(R_i - hits), axis=0) # Shape (n_features,)

            diff_miss_sum = np.zeros(n_features)
            
            for cls in unique_classes:
                if cls == class_R:
                    continue
                
                misses = misses_by_class.get(cls)
                if misses is None or len(misses) == 0:
                    continue

                P_cls = (np.sum(y == cls) / n_samples)
                P_ref = (np.sum(y == class_R) / n_samples)
                # Normalizing probability factor
                P_factor = P_cls / (1 - P_ref + 1e-10)

                diff_miss = np.mean(np.abs(R_i - misses), axis=0)
                diff_miss_sum += P_factor * diff_miss

            weights += diff_miss_sum - diff_hit

        weights /= n_iterations
        # weights[weights < 0] = 0 # ReliefF weights can be negative (irrelevant/redundant)
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

    # ---------------------------------------------------------
    #   DATA AUGMENTATION
    # ---------------------------------------------------------
    @staticmethod
    def augment_audio(audio, sr):
        """
        Applies random augmentations: Noise injection, Time shifting.
        """
        # 1. Noise Injection
        if random.random() < 0.5:
            noise_amp = 0.005 * np.random.uniform() * np.amax(audio)
            audio = audio + noise_amp * np.random.normal(size=audio.shape[0])
            
        # 2. Time Shifting
        if random.random() < 0.5:
            shift_amt = int(random.random() * sr * 0.5) # up to 0.5 sec
            direction = random.choice([-1, 1])
            audio = np.roll(audio, direction * shift_amt)
            
        return audio