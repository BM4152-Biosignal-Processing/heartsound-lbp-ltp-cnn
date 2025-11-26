# Heart Sound Classification

This project implements a heart sound classification system using **1D Local Binary Patterns (1D-LBP)**, **1D Local Ternary Patterns (1D-LTP)**, and a **1D Convolutional Neural Network (CNN)**. It is based on the research paper *"Heart sounds classification using CNN with 1D-LBP and 1D-LTP features"* by Er, Mehmet Bilal.

## Features

*   **Feature Extraction**: Uses 1D-LBP and 1D-LTP for robust texture feature extraction from audio signals.
*   **Feature Selection**: Implements ReliefF algorithm to select the most relevant features.
*   **Classification**: Uses a 1D-CNN to classify heart sounds into categories.
*   **Multiple Datasets**: Supports both **PASCAL** and **Physionet2016** datasets.

## Datasets

The project supports the following datasets:

1.  **PASCAL Classifying Heart Sounds Challenge 2011**
    *   Classes: Normal, Murmur, Artifact, Extrahls
2.  **PhysioNet/Computing in Cardiology Challenge 2016**
    *   Classes: Normal, Abnormal

## Usage

### Training the Model

You can train the model using either the PASCAL or Physionet dataset.

**Option 1: Using `main.py` (Interactive or Command Line)**

Run interactively to select the dataset:
```bash
python main.py
```

Or specify arguments directly:
```bash
# Train on PASCAL dataset
python main.py --dataset PASCAL --epochs 140

# Train on Physionet dataset
python main.py --dataset Physionet --epochs 140
```

**Option 2: Using `train.py` (Advanced)**

```bash
# Train on PASCAL with 10-fold CV
python train.py --mode kfold --dataset PASCAL --epochs 140

# Train on Physionet with 10-fold CV
python train.py --mode kfold --dataset Physionet --epochs 140

# Quick simple training (no CV)
python train.py --mode simple --dataset PASCAL
```

### Making Predictions

To classify a heart sound recording (WAV file):

```bash
python predict.py path/to/your/audio_file.wav
```

Example:
```bash
python predict.py PASCAL/Atraining_normal/201101070538.wav
```

## Project Structure

*   `data_preprocessing.py`: Handles feature extraction (LBP, LTP) and ReliefF selection.
*   `model.py`: Defines the 1D-CNN model architecture.
*   `train.py`: Core training logic.
*   `main.py`: Main entry point with dataset selection.
*   `predict.py`: Script for making predictions on new audio files.
*   `config.py`: Configuration settings.
*   `PASCAL/`: Directory for PASCAL dataset.
*   `Physionet2016/`: Directory for Physionet dataset.

## Citation

If you use this work, please cite the original paper:

> Er, Mehmet Bilal. "Heart sounds classification using CNN with 1D-LBP and 1D-LTP features." *Applied Acoustics* 180 (2021): 108152.
