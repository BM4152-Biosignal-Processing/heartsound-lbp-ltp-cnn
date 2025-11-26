"""
Training script for Heart Sound Classification using 1D-LBP and 1D-LTP features
Based on: "Heart sounds classification using CNN with 1D-LBP and 1D-LTP features"
Er, Mehmet Bilal - Applied Acoustics 2021

Uses 10-fold cross-validation as specified in the paper
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import tensorflow as tf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import (
    EPOCHS, BATCH_SIZE, CLASSES, MODEL_SAVE_PATH, 
    RANDOM_SEED, NUM_CLASSES, N_FOLDS
)
K_FOLDS = N_FOLDS  # Alias for clarity
from data_preprocessing import load_dataset
from model import get_model, get_callbacks


def set_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_training_history(history, fold=None, save_path=None):
    """
    Plot training history
    
    Args:
        history: Keras training history
        fold: Fold number (for k-fold CV)
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    title_suffix = f' (Fold {fold})' if fold else ''
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'Model Accuracy{title_suffix}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'Model Loss{title_suffix}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Path to save the plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f'{title} (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title(f'{title} (Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_cv_results(fold_metrics, save_path=None):
    """
    Plot cross-validation results
    
    Args:
        fold_metrics: Dictionary with metrics for each fold
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    folds = range(1, len(fold_metrics['accuracy']) + 1)
    
    # Accuracy
    axes[0, 0].bar(folds, fold_metrics['accuracy'], color='steelblue', alpha=0.7)
    axes[0, 0].axhline(y=np.mean(fold_metrics['accuracy']), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(fold_metrics["accuracy"]):.4f}')
    axes[0, 0].set_title('Accuracy per Fold')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])
    
    # Precision
    axes[0, 1].bar(folds, fold_metrics['precision'], color='forestgreen', alpha=0.7)
    axes[0, 1].axhline(y=np.mean(fold_metrics['precision']), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(fold_metrics["precision"]):.4f}')
    axes[0, 1].set_title('Precision per Fold')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    
    # Recall
    axes[1, 0].bar(folds, fold_metrics['recall'], color='darkorange', alpha=0.7)
    axes[1, 0].axhline(y=np.mean(fold_metrics['recall']), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(fold_metrics["recall"]):.4f}')
    axes[1, 0].set_title('Recall per Fold')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    
    # F1 Score
    axes[1, 1].bar(folds, fold_metrics['f1'], color='purple', alpha=0.7)
    axes[1, 1].axhline(y=np.mean(fold_metrics['f1']), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(fold_metrics["f1"]):.4f}')
    axes[1, 1].set_title('F1 Score per Fold')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"CV results plot saved to {save_path}")
    
    plt.close()


def train_with_kfold(model_type='paper', dataset_name='PASCAL', epochs=EPOCHS, batch_size=BATCH_SIZE, 
                     k_folds=K_FOLDS):
    """
    Train with K-Fold Cross-Validation as specified in the paper
    
    Args:
        model_type: Type of model ('paper', 'bn', 'adaptive')
        epochs: Number of training epochs
        batch_size: Batch size
        k_folds: Number of folds for cross-validation
    
    Returns:
        Best model and metrics
    """
    import pickle
    
    set_seeds()
    
    print("="*60)
    print(f"HEART SOUND CLASSIFICATION ({dataset_name}) - 1D-LBP/LTP + CNN")
    print("Based on: Er, Mehmet Bilal - Applied Acoustics 2021")
    print("="*60)
    print(f"Model: 1D-CNN ({model_type})")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"K-Folds: {k_folds}")
    print("="*60)
    
    # Load and prepare data
    print("\n[1/3] Extracting 1D-LBP and 1D-LTP features...")
    X, y, selected_indices = load_dataset(dataset_name=dataset_name, augment=True)  # Enable augmentation
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Save feature indices for inference
    if selected_indices is not None:
        indices_path = os.path.join(MODEL_SAVE_PATH, 'feature_indices.pkl')
        with open(indices_path, 'wb') as f:
            pickle.dump(selected_indices, f)
        print(f"Saved {len(selected_indices)} feature indices to {indices_path}")
    
    # Fit and save scaler on full dataset
    full_scaler = StandardScaler()
    full_scaler.fit(X)
    scaler_path = os.path.join(MODEL_SAVE_PATH, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(full_scaler, f)
    print(f"Saved scaler to {scaler_path}")
    
    # Initialize k-fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Store metrics for each fold
    fold_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 
        'loss': [], 'val_accuracy': []
    }
    all_y_true = []
    all_y_pred = []
    
    best_model = None
    best_accuracy = 0
    
    # Results directory
    results_dir = os.path.join(MODEL_SAVE_PATH, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n[2/3] Starting {k_folds}-Fold Cross-Validation...")
    print("-" * 60)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*20} FOLD {fold}/{k_folds} {'='*20}")
        
        # Split data
        X_train, X_val = X[train_idx].copy(), X[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Normalize features (fit on train, transform on val)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Reshape for Conv1D: (samples, features, 1)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Val set: {X_val.shape[0]} samples")
        
        # Get input shape
        input_shape = (X_train.shape[1], 1)
        
        # Build model (fresh model for each fold)
        model = get_model(model_type, input_shape)
        
        # Get callbacks (with fold-specific checkpoint)
        fold_checkpoint = os.path.join(MODEL_SAVE_PATH, f'fold_{fold}_best.keras')
        callbacks = get_callbacks(checkpoint_path=fold_checkpoint, patience=30)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history for this fold
        plot_training_history(
            history, fold=fold,
            save_path=os.path.join(results_dir, f'fold_{fold}_history.png')
        )
        
        # Evaluate on validation set
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        fold_metrics['accuracy'].append(acc)
        fold_metrics['precision'].append(prec)
        fold_metrics['recall'].append(rec)
        fold_metrics['f1'].append(f1)
        fold_metrics['val_accuracy'].append(max(history.history['val_accuracy']))
        
        # Collect predictions for overall confusion matrix
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        
        print(f"\nFold {fold} Results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        # Keep best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            print(f"  *** New best model (Acc: {acc:.4f}) ***")
        
        # Clear session to free memory
        tf.keras.backend.clear_session()
    
    # Summary
    print("\n" + "="*60)
    print("[3/3] CROSS-VALIDATION RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nAccuracy:  {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
    print(f"Precision: {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
    print(f"Recall:    {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
    print(f"F1 Score:  {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
    
    # Plot CV results
    plot_cv_results(fold_metrics, save_path=os.path.join(results_dir, 'cv_results.png'))
    
    # Plot overall confusion matrix
    from config import DATASET_CONFIG
    classes = DATASET_CONFIG[dataset_name]['classes']
    plot_confusion_matrix(
        all_y_true, all_y_pred, classes,
        save_path=os.path.join(results_dir, 'confusion_matrix_overall.png'),
        title='Overall Confusion Matrix (All Folds)'
    )
    
    # Classification report
    print("\nOverall Classification Report:")
    print("-" * 50)
    print(classification_report(all_y_true, all_y_pred, target_names=classes))
    
    # Save best model
    if best_model is not None:
        best_model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.keras')
        best_model.save(best_model_path)
        print(f"\nBest model saved to {best_model_path}")
    
    return best_model, fold_metrics


def train_simple(model_type='paper', dataset_name='PASCAL', epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Simple train/test split training (no cross-validation)
    Faster alternative for quick experiments
    
    Args:
        model_type: Type of model ('paper', 'bn', 'adaptive')
        epochs: Number of training epochs
        batch_size: Batch size
    
    Returns:
        Trained model, history, and metrics
    """
    from sklearn.model_selection import train_test_split
    
    set_seeds()
    
    print("="*60)
    print(f"HEART SOUND CLASSIFICATION ({dataset_name}) - Simple Train/Test Split")
    print("="*60)
    
    # Load and prepare data
    print("\n[1/4] Extracting 1D-LBP and 1D-LTP features...")
    X, y, _ = load_dataset(dataset_name=dataset_name, augment=True)  # Enable augmentation
    
    print(f"\nDataset shape: {X.shape}")
    
    # Split data
    print("\n[2/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_SEED
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Reshape for Conv1D
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Build model
    print(f"\n[3/4] Building {model_type} model...")
    input_shape = (X_train.shape[1], 1)
    model = get_model(model_type, input_shape)
    model.summary()
    
    # Train
    print(f"\n[4/4] Training for {epochs} epochs...")
    callbacks = get_callbacks()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nClassification Report:")
    from config import DATASET_CONFIG
    classes = DATASET_CONFIG[dataset_name]['classes']
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Results directory
    results_dir = os.path.join(MODEL_SAVE_PATH, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot results
    plot_training_history(history, save_path=os.path.join(results_dir, 'training_history.png'))
    plot_confusion_matrix(y_test, y_pred, classes, 
                          save_path=os.path.join(results_dir, 'confusion_matrix.png'))
    
    # Save model
    model_path = os.path.join(MODEL_SAVE_PATH, 'final_model.keras')
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    
    return model, history, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Heart Sound Classification Model')
    parser.add_argument('--mode', type=str, default='kfold', choices=['kfold', 'simple'],
                        help='Training mode: kfold (10-fold CV) or simple (train/test split)')
    parser.add_argument('--model', type=str, default='adaptive', choices=['paper', 'bn', 'adaptive'],
                        help='Model type: paper (exact architecture), bn (with batch norm), adaptive')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--dataset', type=str, default='PASCAL', choices=['PASCAL', 'Physionet'],
                        help='Dataset to use: PASCAL or Physionet')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Heart Sound Classification using 1D-LBP/LTP + 1D-CNN")
    print("="*60)
    
    if args.mode == 'kfold':
        # 10-Fold Cross-Validation (as in paper)
        best_model, fold_metrics = train_with_kfold(
            model_type=args.model,
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            k_folds=K_FOLDS
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Mean Accuracy: {np.mean(fold_metrics['accuracy']):.4f}")
        print(f"Mean F1 Score: {np.mean(fold_metrics['f1']):.4f}")
        
    else:
        # Simple train/test split
        model, history, metrics = train_simple(
            model_type=args.model,
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {metrics['f1']:.4f}")
