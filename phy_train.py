import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import sys


# --- Configuration ---
N_SELECTED_FEATURES = 100 
N_CLASSES = 2  # <--- BINARY CLASSIFICATION
RANDOM_SEED = 42
EPOCHS = 100
BATCH_SIZE = 64

# Define the class names
CLASS_NAMES = ["Healthy (Normal)", "Unhealthy (Abnormal)"] 

def load_data(X_file, y_file):
    """Loads feature data and labels from CSV files."""
    try:
        X = np.loadtxt(X_file, delimiter=",")
        y = np.loadtxt(y_file, delimiter=",", dtype=int)
        print(f"✅ Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except FileNotFoundError as e:
        print(f"❌ Error: Required feature file not found. Ensure main_physionet.py was run successfully.")
        sys.exit(1)
        return None, None

def create_cnn_model(input_length, num_classes):
    """
    Implements the 1D CNN architecture for Binary Classification.
    """
    model = Sequential([
        # Input: (100, 1)

        # ----------------- Block 1 -----------------
        Conv1D(filters=64, kernel_size=1, strides=1, activation='relu', input_shape=(input_length, 1)),
        MaxPooling1D(pool_size=2, strides=2),
        
        # ----------------- Block 2 -----------------
        Conv1D(filters=32, kernel_size=1, strides=1, activation='relu'),
        MaxPooling1D(pool_size=2, strides=2),

        # ----------------- Block 3 -----------------
        Conv1D(filters=32, kernel_size=1, strides=1, activation='relu'),
        MaxPooling1D(pool_size=2, strides=2),

        # ----------------- Block 4 -----------------
        Conv1D(filters=16, kernel_size=1, strides=1, activation='relu'),
        MaxPooling1D(pool_size=2, strides=2),

        # ----------------- Fully Connected Layers -----------------
        Flatten(), 
        Dense(64, activation='relu'), 
        Dropout(0.5),                 
        Dense(32, activation='relu'), 
        
        # Output Layer: 2 classes
        Dense(num_classes, activation='softmax') 
    ])
    return model

# ----------------------------------------------------------------------
#                             MAIN EXECUTION
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Load Data: LOADING THE NEW FOLDER-BASED FILES
    X_reduced, y_dataset = load_data("outputs/X_reduced_PN2016_2class_folder.csv", "outputs/y_dataset_PN2016_2class_folder.csv")

    # 2. Reshape and Encode Data
    X_reshaped = X_reduced.reshape(X_reduced.shape[0], N_SELECTED_FEATURES, 1)
    y_encoded = to_categorical(y_dataset, num_classes=N_CLASSES)

    # 3. Split Data (Stratify ensures equal proportion of classes in train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_encoded, test_size=0.3, random_state=RANDOM_SEED, stratify=y_dataset
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")


    # 4. Handle Class Imbalance (CRUCIAL for PhysioNet)
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_dataset), 
        y=y_dataset
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("\nCalculated Class Weights (to address imbalance):", class_weight_dict)

    # 5. Create and Compile Model
    model = create_cnn_model(N_SELECTED_FEATURES, N_CLASSES)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n--- Model Summary ---")
    model.summary()

    # 6. Train the Model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,            
        batch_size=BATCH_SIZE,         
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,  # Applying class weights
        verbose=1              
    )

    # 7. Evaluate Model on Test Set
    print("\n--- Final Model Evaluation ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")