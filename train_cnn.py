import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Must match the size of X_reduced.csv
N_SELECTED_FEATURES = 100 
N_CLASSES = 4 
RANDOM_SEED = 42

# Define the class names based on your previous mapping (for clear output)
CLASS_NAMES = ["Normal", "Murmur", "ExtraHLS", "Artifact"] 

def load_data(X_file, y_file):
    """Loads feature data and labels from CSV files."""
    try:
        X = np.loadtxt(X_file, delimiter=",")
        y = np.loadtxt(y_file, delimiter=",", dtype=int)
        print(f"✅ Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found. Ensure you ran the ReliefF script first.")
        print(f"Missing file: {e.filename}")
        return None, None

def create_cnn_model(input_length, num_classes):
    """
    Implements the 1D CNN architecture based on the paper (Table 1).
    Input Shape: (input_length, 1)
    """
    model = Sequential([
        # ----------------- Block 1 -----------------
        # Conv Lyer 1: 64x1 kernel 1, stride 1
        Conv1D(filters=64, kernel_size=1, strides=1, activation='relu', input_shape=(input_length, 1)),
        MaxPooling1D(pool_size=2, strides=2),  # Output length: 100 -> 50

        # ----------------- Block 2 -----------------
        # Conv Lyer 2: 32x1 kernel 1, stride 1
        Conv1D(filters=32, kernel_size=1, strides=1, activation='relu'),
        MaxPooling1D(pool_size=2, strides=2),  # Output length: 50 -> 25

        # ----------------- Block 3 -----------------
        # Conv Lyer 3: 32x1 kernel 1, stride 1
        Conv1D(filters=32, kernel_size=1, strides=1, activation='relu'),
        MaxPooling1D(pool_size=2, strides=2),  # Output length: 25 -> 12 (due to truncation)

        # ----------------- Block 4 -----------------
        # Conv Lyer 4: 16x1 kernel 1, stride 1
        Conv1D(filters=16, kernel_size=1, strides=1, activation='relu'),
        MaxPooling1D(pool_size=2, strides=2),  # Output length: 12 -> 6

        # ----------------- Fully Connected Layers -----------------
        Flatten(), # Flattened size: 6 * 16 = 96 units

        Dense(64, activation='relu'), # Fc_1: 64 neurons
        Dropout(0.5),                 # Drop_1: 50% dropout rate
        Dense(32, activation='relu'), # Fc_2: 32 neurons
        
        # Output Layer: Softmax for 4-class classification
        Dense(num_classes, activation='softmax') 
    ])
    return model

# ----------------------------------------------------------------------
#                             MAIN EXECUTION
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Load Data
    X_reduced, y_dataset = load_data("outputs/X_reduced.csv", "outputs/y_dataset.csv")

    if X_reduced is None:
        exit() # Exit if loading failed

    # 2. Reshape and Encode Data
    
    # CNNs expect a 3D input: (samples, time_steps/length, channels)
    # Our data is (samples, 100). We add a channel dimension of 1.
    X_reshaped = X_reduced.reshape(X_reduced.shape[0], N_SELECTED_FEATURES, 1)
    
    # Convert integer labels to one-hot encoding (e.g., 0 -> [1, 0, 0, 0])
    y_encoded = to_categorical(y_dataset, num_classes=N_CLASSES)

    print(f"Data prepared. X_reshaped shape: {X_reshaped.shape}")
    print(f"y_encoded shape: {y_encoded.shape}")

    # 3. Split Data into Training and Testing Sets
    # A standard 70/30 split is often used in this domain
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_encoded, test_size=0.3, random_state=RANDOM_SEED, stratify=y_encoded
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # 4. Create and Compile Model
    model = create_cnn_model(N_SELECTED_FEATURES, N_CLASSES)

    # Use common settings for multi-class classification:
    # Optimizer: Adam (efficient default), Loss: Categorical Cross-Entropy, Metric: Accuracy
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n--- Model Summary ---")
    model.summary()

    # 5. Train the Model
    print("\n--- Starting Model Training ---")
    # Training parameters often found in similar papers
    history = model.fit(
        X_train, y_train,
        epochs=100,            # Number of training iterations (may need tuning)
        batch_size=32,         # Size of data chunks processed at once
        validation_data=(X_test, y_test),
        verbose=1              # Show training progress
    )

    # 6. Evaluate Model on Test Set
    print("\n--- Final Model Evaluation ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Optional: Save the trained model
    # model.save("outputs/cnn_model.h5")
    # print("\nModel saved to outputs/cnn_model.h5")