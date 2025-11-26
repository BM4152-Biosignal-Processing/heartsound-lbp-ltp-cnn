"""
1D-CNN Model Architecture for Heart Sound Classification
Based on: "Heart sounds classification using CNN with 1D-LBP and 1D-LTP features"
Er, Mehmet Bilal - Applied Acoustics 2021

Architecture from Table 1:
- 4 Convolution layers (64, 32, 32, 16 filters)
- 4 MaxPooling layers (2x1, stride 2)
- 2 Fully connected layers (64, 32 neurons)
- Dropout 50%
- Softmax output
"""
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    Dense, Flatten, GlobalAveragePooling1D, Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
import os

from config import NUM_CLASSES, LEARNING_RATE, MODEL_SAVE_PATH, CHECKPOINT_PATH, DROPOUT_RATE


def build_1d_cnn_paper(input_shape, num_classes=NUM_CLASSES):
    """
    Build 1D-CNN exactly as described in paper Table 1
    
    Architecture:
    1. Input: LBP + LTP features
    2. Conv1D (64 filters, kernel 64x1, stride 1) + ReLU
    3. MaxPooling1D (2x1, stride 2)
    4. Conv1D (32 filters, kernel 32x1, stride 1) + ReLU
    5. MaxPooling1D (2x1, stride 2)
    6. Conv1D (32 filters, kernel 32x1, stride 1) + ReLU
    7. MaxPooling1D (2x1, stride 2)
    8. Conv1D (16 filters, kernel 16x1, stride 1) + ReLU
    9. MaxPooling1D (2x1, stride 2)
    10. Flatten
    11. FC (64 neurons) + ReLU
    12. Dropout (50%)
    13. FC (32 neurons) + ReLU
    14. Softmax output
    
    Args:
        input_shape: Shape of input features (n_features, 1)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape, name='input_lbp_ltp_features')
    
    # Convolution1D_1: 64 filters, kernel size 64
    x = Conv1D(filters=64, kernel_size=64, strides=1, padding='same', name='conv1d_1')(inputs)
    x = Activation('relu', name='relu_1')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool_1')(x)
    
    # Convolution1D_2: 32 filters, kernel size 32
    x = Conv1D(filters=32, kernel_size=32, strides=1, padding='same', name='conv1d_2')(x)
    x = Activation('relu', name='relu_2')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool_2')(x)
    
    # Convolution1D_3: 32 filters, kernel size 32
    x = Conv1D(filters=32, kernel_size=32, strides=1, padding='same', name='conv1d_3')(x)
    x = Activation('relu', name='relu_3')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool_3')(x)
    
    # Convolution1D_4: 16 filters, kernel size 16
    x = Conv1D(filters=16, kernel_size=16, strides=1, padding='same', name='conv1d_4')(x)
    x = Activation('relu', name='relu_4')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool_4')(x)
    
    # Flatten
    x = Flatten(name='flatten')(x)
    
    # Fc_1: 64 neurons
    x = Dense(64, name='fc_1')(x)
    x = Activation('relu', name='relu_5')(x)
    
    # Dropout: 50%
    x = Dropout(DROPOUT_RATE, name='dropout')(x)
    
    # Fc_2: 32 neurons
    x = Dense(32, name='fc_2')(x)
    x = Activation('relu', name='relu_6')(x)
    
    # Output: Softmax
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='HeartSound_1D_CNN')
    
    return model


def build_1d_cnn_with_bn(input_shape, num_classes=NUM_CLASSES):
    """
    Build 1D-CNN with Batch Normalization for better training stability
    
    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv1D(64, kernel_size=64, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    # Block 2
    x = Conv1D(32, kernel_size=32, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    # Block 3
    x = Conv1D(32, kernel_size=32, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    # Block 4
    x = Conv1D(16, kernel_size=16, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    # Flatten and Dense
    x = Flatten()(x)
    
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='HeartSound_1D_CNN_BN')
    
    return model


def build_1d_cnn_adaptive(input_shape, num_classes=NUM_CLASSES):
    """
    Build adaptive 1D-CNN that adjusts kernel sizes based on input
    This handles smaller feature dimensions
    
    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    n_features = input_shape[0]
    
    # Adaptive kernel sizes (max 1/4 of feature dimension)
    k1 = min(64, n_features // 4)
    k2 = min(32, n_features // 8)
    k3 = min(32, n_features // 8)
    k4 = min(16, n_features // 16)
    
    # Ensure minimum kernel size
    k1 = max(3, k1)
    k2 = max(3, k2)
    k3 = max(3, k3)
    k4 = max(3, k4)
    
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv1D(64, kernel_size=k1, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    
    # Block 2
    x = Conv1D(32, kernel_size=k2, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    
    # Block 3
    x = Conv1D(32, kernel_size=k3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    
    # Block 4
    x = Conv1D(16, kernel_size=k4, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    
    # Global pooling instead of flatten (more robust)
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(32, activation='relu')(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='HeartSound_1D_CNN_Adaptive')
    
    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile model with optimizer, loss, and metrics
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # Using integer labels
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(checkpoint_path=CHECKPOINT_PATH, patience=30):
    """
    Get training callbacks
    
    Args:
        checkpoint_path: Path to save model checkpoints
        patience: Patience for early stopping
    
    Returns:
        List of callbacks
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def get_model(model_type='paper', input_shape=None, num_classes=NUM_CLASSES):
    """
    Factory function to get model by type
    
    Args:
        model_type: Type of model ('paper', 'bn', 'adaptive')
        input_shape: Input shape for the model
        num_classes: Number of output classes
    
    Returns:
        Compiled model
    """
    if model_type == 'paper':
        model = build_1d_cnn_paper(input_shape, num_classes)
    elif model_type == 'bn':
        model = build_1d_cnn_with_bn(input_shape, num_classes)
    elif model_type == 'adaptive':
        model = build_1d_cnn_adaptive(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = compile_model(model)
    
    return model


if __name__ == "__main__":
    # Test model building
    print("Testing 1D-CNN model (paper architecture)...")
    
    # Example input shape: 256 features with 1 channel
    input_shape = (256, 1)
    
    model = build_1d_cnn_paper(input_shape)
    model = compile_model(model)
    model.summary()
    
    print("\n" + "="*50 + "\n")
    
    print("Testing 1D-CNN with Batch Normalization...")
    model_bn = build_1d_cnn_with_bn(input_shape)
    model_bn = compile_model(model_bn)
    model_bn.summary()
