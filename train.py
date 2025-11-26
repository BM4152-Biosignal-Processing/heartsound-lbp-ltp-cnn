import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns1
import os
import sys

# ============================================================
#               CONFIGURATION & USER INPUT
# ============================================================

print("\n======================================================")
print("   Heart Sound Classification - CNN Training")
print("======================================================\n")

print("Select Dataset to Train on:")
print("  [1] PhysioNet2016")
print("  [2] PASCAL")

choice = input("Enter choice (1 or 2): ").strip()

if choice == "1":
    DATASET_NAME = "PhysioNet2016"
    N_CLASSES = 2
    CLASS_NAMES = ["Healthy", "Unhealthy"]
    X_FILE = "outputs/PhysioNet_X_reduced.csv"
    Y_FILE = "outputs/PhysioNet_y.csv"
elif choice == "2":
    DATASET_NAME = "PASCAL"
    N_CLASSES = 4
    CLASS_NAMES = ["Normal", "Murmur", "ExtraHLS", "Artifact"]
    X_FILE = "outputs/PASCAL_X_reduced.csv"
    Y_FILE = "outputs/PASCAL_y.csv" # Note: main.py saves as _y.csv now
else:
    print("[ERROR] Invalid choice. Exiting.")
    sys.exit(1)

RANDOM_SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ============================================================
#                     DATA LOADING
# ============================================================

def load_data():
    """Loads feature data and labels from CSV files."""
    try:
        if not os.path.exists(X_FILE) or not os.path.exists(Y_FILE):
             # Fallback check for PASCAL_y_dataset.csv if PASCAL_y.csv doesn't exist (backward compatibility)
             if DATASET_NAME == "PASCAL" and os.path.exists("outputs/PASCAL_y_dataset.csv"):
                 y_path = "outputs/PASCAL_y_dataset.csv"
             else:
                 print(f"[ERROR] Data files not found for {DATASET_NAME}.")
                 print(f"   Expected: {X_FILE} and {Y_FILE}")
                 return None, None
        else:
            y_path = Y_FILE

        print(f"Loading features from: {X_FILE}")
        print(f"Loading labels from:   {y_path}")

        X = np.loadtxt(X_FILE, delimiter=",")
        y = np.loadtxt(y_path, delimiter=",", dtype=int)
        
        print(f"[OK] Data loaded for {DATASET_NAME}.")
        print(f"  Features: {X.shape}")
        print(f"  Labels: {y.shape}")
        return X, y
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return None, None

# ============================================================
#                     MODEL ARCHITECTURE
# ============================================================

def create_specific_cnn_model(input_shape, num_classes):
    """
    Implements the CNN architecture from Table 1.
    """
    model = models.Sequential()
    
    # Input Layer (Implicit in first layer)
    # Layer 2: Convolution1D_1 (64, 11, 1)
    model.add(layers.Conv1D(filters=64, kernel_size=11, strides=1, padding='same', activation='relu', input_shape=input_shape))
    # Layer 4: MaxPooling1D_1 (2, 2)
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    
    # Layer 5: Convolution1D_2 (32, 11, 1)
    model.add(layers.Conv1D(filters=32, kernel_size=11, strides=1, padding='same', activation='relu'))
    # Layer 7: MaxPooling1D_2 (2, 2)
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    
    # Layer 8: Convolution1D_3 (32, 11, 1)
    model.add(layers.Conv1D(filters=32, kernel_size=11, strides=1, padding='same', activation='relu'))
    # Layer 10: MaxPooling1D_3 (2, 2)
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    
    # Layer 11: Convolution1D_4 (16, 11, 1)
    model.add(layers.Conv1D(filters=16, kernel_size=11, strides=1, padding='same', activation='relu'))
    # Layer 13: MaxPooling1D_4 (2, 2)
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    
    # Layer 14: Fc_1 (64 neurons)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    
    # Layer 16: Drop_1 (50%)
    model.add(layers.Dropout(0.5))
    
    # Layer 17: Fc_2 (32 neurons)
    model.add(layers.Dense(32, activation='relu'))
    
    # Layer 18: Output (Softmax)
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ============================================================
#                     PLOTTING FUNCTIONS
# ============================================================

def plot_history(history):
    """Plots training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(f"outputs/history_plot_{DATASET_NAME}.png")
    print(f"Saved history plot to outputs/history_plot_{DATASET_NAME}.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {DATASET_NAME}')
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{DATASET_NAME}.png")
    print(f"Saved confusion matrix to outputs/confusion_matrix_{DATASET_NAME}.png")
    plt.close()
    
    # Calculate TP, TN, FP, FN for each class (One-vs-Rest)
    print("\n--- Detailed Metrics (One-vs-Rest) ---")
    for i, class_name in enumerate(classes):
        # Treat class i as Positive, others as Negative
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        print(f"Class '{class_name}':")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        
    # Plot TP, TN, FP, FN graph (Four Boxes Graph) for Binary Classification
    if len(classes) == 2:
        # Aggregate for the positive class (usually index 1 'Unhealthy' or similar, but let's use index 1)
        # Assuming index 1 is the "Positive" class of interest (e.g. Unhealthy/Murmur)
        tp = cm[1, 1]
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        
        labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        counts = [tn, fp, fn, tp]
        
        plt.figure(figsize=(6, 5))
        bars = plt.bar(labels, counts, color=['green', 'red', 'red', 'green'])
        plt.title(f'TP, TN, FP, FN Counts - {DATASET_NAME}')
        plt.ylabel('Count')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')
        plt.tight_layout()
        plt.savefig(f"outputs/tp_tn_fp_fn_plot_{DATASET_NAME}.png")
        print(f"Saved TP/TN/FP/FN plot to outputs/tp_tn_fp_fn_plot_{DATASET_NAME}.png")
        plt.close()

def plot_roc_curve(y_test_encoded, y_pred_probs, n_classes, classes):
    """Plots ROC curve."""
    plt.figure(figsize=(8, 6))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for {classes[i]}')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {DATASET_NAME}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"outputs/roc_curve_{DATASET_NAME}.png")
    print(f"Saved ROC curve to outputs/roc_curve_{DATASET_NAME}.png")
    plt.close()

# ----------------------------------------------------------------------
#                             MAIN EXECUTION
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    print(f"\n=========== Training CNN on {DATASET_NAME} ===========\n")
    
    # 1. Load Data
    X, y = load_data()
    if X is None:
        sys.exit(1)

    # 2. Prepare Data
    # Reshape X for 1D CNN: (N_samples, N_features, 1)
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
        
    n_features = X.shape[1]
    X_reshaped = X.reshape(X.shape[0], n_features, 1)
    
    # One-hot encode labels
    y_encoded = to_categorical(y, num_classes=N_CLASSES)
    
    # 3. Split Data (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_encoded, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    
    # Convert one-hot back to integers for metrics/weights
    y_train_int = np.argmax(y_train, axis=1)
    y_test_int = np.argmax(y_test, axis=1)

    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    
    # 4. Handle Class Imbalance
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train_int), 
        y=y_train_int
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weight_dict}")

    # 5. Create and Train Model
    model = create_specific_cnn_model(input_shape=(n_features, 1), num_classes=N_CLASSES)
    model.summary()
    
    # Callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    print("\n--- Starting Training ---")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )
    
    # 6. Evaluation & Plotting
    print("\n--- Final Evaluation ---")
    
    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    acc = accuracy_score(y_test_int, y_pred)
    prec = precision_score(y_test_int, y_pred, average='weighted')
    rec = recall_score(y_test_int, y_pred, average='weighted') # Sensitivity
    
    print(f"\nAccuracy:    {acc*100:.2f}%")
    print(f"Precision:   {prec*100:.2f}%")
    print(f"Sensitivity: {rec*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test_int, y_pred, target_names=CLASS_NAMES))
    
    # Plots
    plot_history(history)
    plot_confusion_matrix(y_test_int, y_pred, CLASS_NAMES)
    plot_roc_curve(y_test, y_pred_probs, N_CLASSES, CLASS_NAMES)

    # Save Model
    model_save_path = f"outputs/cnn_model_{DATASET_NAME}.h5"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")