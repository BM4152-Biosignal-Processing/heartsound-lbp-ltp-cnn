import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
DATASET_NAME = "PhysioNet2016" # Change to "PASCAL" to train on PASCAL data
# DATASET_NAME = "PASCAL"

if DATASET_NAME == "PhysioNet2016":
    N_CLASSES = 2
    CLASS_NAMES = ["Healthy", "Unhealthy"]
elif DATASET_NAME == "PASCAL":
    N_CLASSES = 4
    CLASS_NAMES = ["Normal", "Murmur", "ExtraHLS", "Artifact"]

RANDOM_SEED = 42

def load_data():
    """Loads feature data and labels from NPY files."""
    try:
        prefix = DATASET_NAME
        X_spec = np.load(f"outputs/{prefix}_X_spectrogram.npy")
        X_mfcc = np.load(f"outputs/{prefix}_X_mfcc.npy")
        y = np.load(f"outputs/{prefix}_y_dataset.npy")
        print(f"✅ Data loaded for {DATASET_NAME}.")
        print(f"  Spectrograms: {X_spec.shape}")
        print(f"  MFCCs: {X_mfcc.shape}")
        print(f"  Labels: {y.shape}")
        return X_spec, X_mfcc, y
    except FileNotFoundError as e:
        print(f"❌ Error: Data file not found for {DATASET_NAME}. Run main_physionet.py with correct dataset selected.")
        return None, None, None

def create_2d_cnn_model(input_shape, num_classes):
    """
    Implements the 2D CNN architecture for Spectrograms.
    Refined for higher accuracy (BatchNormalization, Dropout, More Filters).
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 4 (Added for deeper feature extraction)
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Using a lower learning rate for stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ----------------------------------------------------------------------
#                             MAIN EXECUTION
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Load Data
    X_spec, X_mfcc, y = load_data()
    if X_spec is None:
        exit()

    # 2. Prepare Data
    # One-hot encode labels for CNN
    y_encoded = to_categorical(y, num_classes=N_CLASSES)
    
    # 3. Split Data (Stratified)
    # We need to split both Spectrograms and MFCCs with the same indices
    X_spec_train, X_spec_test, X_mfcc_train, X_mfcc_test, y_train, y_test = train_test_split(
        X_spec, X_mfcc, y_encoded, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    
    # Convert one-hot back to integers for SVM
    y_train_int = np.argmax(y_train, axis=1)
    y_test_int = np.argmax(y_test, axis=1)

    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

    # ---------------------------------------------------------
    #   A. Train CNN (Spectrograms)
    # ---------------------------------------------------------
    print("\n--- Training CNN (Spectrograms) ---")
    cnn_model = create_2d_cnn_model(input_shape=X_spec.shape[1:], num_classes=N_CLASSES)
    
    # Callbacks for better training
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    history = cnn_model.fit(
        X_spec_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_spec_test, y_test),
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )
    
    # Get CNN Probabilities
    cnn_probs_test = cnn_model.predict(X_spec_test)
    cnn_pred_test = np.argmax(cnn_probs_test, axis=1)
    acc_cnn = accuracy_score(y_test_int, cnn_pred_test)
    print(f"CNN Accuracy: {acc_cnn:.4f}")

    # ---------------------------------------------------------
    #   B. Train SVM (MFCCs)
    # ---------------------------------------------------------
    print("\n--- Training SVM (MFCCs) ---")
    # Scale MFCC features
    scaler = StandardScaler()
    X_mfcc_train_scaled = scaler.fit_transform(X_mfcc_train)
    X_mfcc_test_scaled = scaler.transform(X_mfcc_test)
    
    # Tuned SVM
    svm_model = SVC(C=10, kernel='rbf', probability=True, random_state=RANDOM_SEED)
    svm_model.fit(X_mfcc_train_scaled, y_train_int)
    
    # Get SVM Probabilities
    svm_probs_test = svm_model.predict_proba(X_mfcc_test_scaled)
    svm_pred_test = np.argmax(svm_probs_test, axis=1)
    acc_svm = accuracy_score(y_test_int, svm_pred_test)
    print(f"SVM Accuracy: {acc_svm:.4f}")

    # ---------------------------------------------------------
    #   C. Ensemble Fusion
    # ---------------------------------------------------------
    print("\n--- Ensemble Fusion (Weighted Average) ---")
    
    # Fusion Rule: P_final = alpha * P_cnn + (1 - alpha) * P_svm
    alpha = 0.6 
    
    final_probs = (alpha * cnn_probs_test) + ((1 - alpha) * svm_probs_test)
    final_pred = np.argmax(final_probs, axis=1)
    
    acc_fusion = accuracy_score(y_test_int, final_pred)
    print(f"Fusion Accuracy: {acc_fusion:.4f}")
    
    print("\nClassification Report (Fusion):")
    print(classification_report(y_test_int, final_pred, target_names=CLASS_NAMES))
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