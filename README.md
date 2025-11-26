# Heart Sound Classification using 1D-LBP, 1D-LTP and CNN

Implementation of **"Heart sounds classification using CNN with 1D-LBP and 1D-LTP features"** by Er, Mehmet Bilal (Applied Acoustics, 2021).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🎯 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 90.14% ± 3.03% |
| **Precision** | 90.48% ± 2.92% |
| **Recall** | 90.14% ± 3.03% |
| **F1 Score** | 90.16% ± 2.99% |

*Evaluated using 10-fold cross-validation on PASCAL Heart Sound dataset*

## 📖 Overview

This project classifies heart sounds into four categories using a novel approach based on:
- **1D Local Binary Patterns (1D-LBP)** for texture feature extraction
- **1D Local Ternary Patterns (1D-LTP)** for robust feature representation
- **ReliefF** feature selection algorithm
- **1D Convolutional Neural Network** for classification

### Classes
| Class | Description | Samples |
|-------|-------------|---------|
| **Normal** | Normal heart sounds (S1-S2) | 31 |
| **Murmur** | Heart sounds with murmurs | 34 |
| **Artifact** | Recordings with noise/artifacts | 40 |
| **Extrahls** | Extra heart sounds (S3/S4) | 19 |

## 🏗️ Architecture

```
Audio Signal
     │
     ▼
┌─────────────────────────────────────┐
│  Butterworth Bandpass Filter        │
│  (5th order, 25-400 Hz)             │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Multi-scale Segmentation           │
│  (1, 3, 9 segments)                 │
└─────────────────────────────────────┘
     │
     ├──────────────┬──────────────┐
     ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ 1D-LBP  │   │ 1D-LTP  │   │ 1D-LTP  │
│(8 neigh)│   │   UP    │   │   LP    │
│ 256 bins│   │ 256 bins│   │ 256 bins│
└─────────┘   └─────────┘   └─────────┘
     │              │              │
     └──────────────┴──────────────┘
                    │
                    ▼
         2304 total features
                    │
                    ▼
┌─────────────────────────────────────┐
│  ReliefF Feature Selection          │
│  (Select top 256 features)          │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│  1D-CNN Classifier                  │
│  4 Conv1D → GlobalAvgPool → Dense   │
└─────────────────────────────────────┘
                    │
                    ▼
        4-class prediction
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YasiruDEX/Heart_sounds-2.git
cd Heart_sounds-2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Make Predictions

```bash
# Predict on a heart sound recording
python predict.py path/to/heart_sound.wav

# Example with PASCAL dataset
python predict.py PASCAL/Atraining_normal/201101070538.wav
```

**Sample Output:**
```
============================================================
Analyzing: heart_sound.wav
============================================================

🔊 PREDICTION: NORMAL
📊 Confidence: 99.00%

📈 All class probabilities:
----------------------------------------
  normal      :  99.00% █████████████████████████████ ◄
  extrahls    :   0.84% 
  murmur      :   0.15% 
  artifact    :   0.00% 

📋 Clinical Interpretation:
  ✅ Normal heart sounds detected (S1-S2)
```

### Training

```bash
# Train with 10-fold cross-validation (recommended, as in paper)
python train.py --mode kfold --model adaptive --epochs 140

# Quick training with train/test split
python train.py --mode simple --epochs 100
```

## 📁 Project Structure

```
Heart_sounds-2/
├── PASCAL/                      # Dataset directory
│   ├── Atraining_normal/        # Normal heart sounds
│   ├── Atraining_murmur/        # Murmur heart sounds
│   ├── Atraining_artifact/      # Artifact recordings
│   └── Atraining_extrahls/      # Extra heart sounds
├── models/                      # Trained models
│   ├── best_model.keras         # Best performing model ⭐
│   ├── scaler.pkl               # Feature scaler
│   ├── feature_indices.pkl      # Selected feature indices
│   ├── fold_*_best.keras        # Best model per fold
│   └── results/                 # Training visualizations
│       ├── cv_results.png
│       ├── confusion_matrix_overall.png
│       └── fold_*_history.png
├── config.py                    # Configuration parameters
├── data_preprocessing.py        # 1D-LBP, 1D-LTP, ReliefF
├── model.py                     # 1D-CNN architecture
├── train.py                     # Training script
├── predict.py                   # Inference script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## ⚙️ Configuration

Key parameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SAMPLE_RATE` | 4000 Hz | Audio sample rate |
| `DURATION` | 5 sec | Audio clip duration |
| `BUTTER_ORDER` | 5 | Butterworth filter order |
| `BUTTER_LOW` | 25 Hz | Low cutoff frequency |
| `BUTTER_HIGH` | 400 Hz | High cutoff frequency |
| `LBP_NEIGHBORS` | 8 | LBP neighborhood size |
| `LTP_THRESHOLD` | 0.02 | LTP threshold |
| `N_FEATURES_TO_SELECT` | 256 | ReliefF selected features |
| `N_FOLDS` | 10 | Cross-validation folds |
| `EPOCHS` | 140 | Training epochs |
| `BATCH_SIZE` | 64 | Batch size |

## 📊 Model Details

### 1D-CNN Architecture

| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| Input | (256, 1) | - |
| Conv1D (64 filters) | (256, 64) | 256 |
| BatchNorm + MaxPool | (128, 64) | 256 |
| Conv1D (32 filters) | (128, 32) | 6,176 |
| BatchNorm + MaxPool | (64, 32) | 128 |
| Conv1D (32 filters) | (64, 32) | 3,104 |
| BatchNorm + MaxPool | (32, 32) | 128 |
| Conv1D (16 filters) | (32, 16) | 1,552 |
| BatchNorm + MaxPool | (16, 16) | 64 |
| GlobalAveragePooling | (16,) | - |
| Dense (64) + Dropout | (64,) | 1,088 |
| Dense (32) + Dropout | (32,) | 2,080 |
| Dense (4, softmax) | (4,) | 132 |

**Total Parameters:** ~14,964

## 🔬 Feature Extraction

### 1D-LBP (Local Binary Pattern)
Compares each sample with its neighbors to create binary pattern:
```
LBP(xc) = Σ s(xi - xc) × 2^i, where s(x) = 1 if x ≥ 0, else 0
```

### 1D-LTP (Local Ternary Pattern)
Three-valued extension of LBP with threshold `t`:
```
s'(x, xc, t) = 1 if x > xc + t
             = 0 if |x - xc| ≤ t  
             = -1 if x < xc - t
```

### ReliefF Feature Selection
Weights features based on their ability to distinguish between classes using nearest neighbor analysis.

## 📈 Training Results

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.84 | 0.87 | 0.85 | 279 |
| Murmur | 0.91 | 0.91 | 0.91 | 306 |
| Artifact | 0.99 | 0.97 | 0.98 | 360 |
| Extrahls | 0.82 | 0.79 | 0.81 | 171 |

### Data Augmentation
8× augmentation per sample:
- Gaussian noise addition
- Time shifting
- Speed perturbation
- Pitch shifting

## 🛠️ Dependencies

```
tensorflow>=2.10.0
numpy<2.0
librosa>=0.9.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 📚 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{er2021heart,
  title={Heart sounds classification using CNN with 1D-LBP and 1D-LTP features},
  author={Er, Mehmet Bilal},
  journal={Applied Acoustics},
  volume={180},
  pages={108152},
  year={2021},
  publisher={Elsevier}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
