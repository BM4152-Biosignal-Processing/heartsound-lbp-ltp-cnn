# heartsound-lbp-ltp-cnn
Implementation of the research paper “Heart Sounds Classification Using a CNN with 1D-Local Binary and Ternary Patterns”. Includes preprocessing, feature extraction (1D-LBP and 1D-LTP), and convolutional neural network–based classification of heart sound signals.

# Heart Sound Classification using LBP, LTP, and 1D CNN

This project implements a heart sound classification pipeline based on 1D Local Binary Patterns (LBP) and Local Ternary Patterns (LTP) features, selected via ReliefF, and classified using a 1D Convolutional Neural Network (CNN).

## Project Structure
- `utils.py`: Utility functions for bandpass filtering, LBP/LTP feature extraction, and ReliefF feature selection.
- `main.py`: Main script for processing the **PASCAL** dataset (Feature Extraction -> Selection -> Saving).
- `main_physionet.py`: Main script for processing the **PhysioNet** dataset.
- `train_cnn.py`: Script for training the 1D CNN model, evaluating performance, and generating plots.
- `verify_pipeline.py`: A simple script to verify the feature extraction logic on synthetic data.

## How to Run (Execution Order)

### 1. Feature Extraction
Run the main script and select your dataset (1 for PhysioNet, 2 for PASCAL):
```bash
python main.py
```
*This will generate feature CSV files in the `outputs/` directory.*

### 2. Model Training
Run the training script and select the same dataset:
```bash
python train.py
```
*This will train the CNN model, print validation metrics, and save plots to `outputs/`.*

## Outputs
The `outputs/` directory will contain:
-   **CSV Files**: Extracted features and labels.
-   **Models**: Saved `.h5` model files.
-   **Plots**:
    -   `history_plot_*.png`: Training/Validation Accuracy and Loss curves.
    -   `confusion_matrix_*.png`: Confusion Matrix heatmap.
    -   `roc_curve_*.png`: ROC Curves for each class.
    -   `tp_tn_fp_fn_plot_*.png`: Bar chart of TP, TN, FP, FN counts (for binary classification).
