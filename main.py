"""
Main entry point for Heart Sound Classification
Run this script to train and evaluate the model
"""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_with_kfold
from config import EPOCHS, BATCH_SIZE, N_FOLDS


def main():
    parser = argparse.ArgumentParser(description='Heart Sound Classification')
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='paper',
        choices=['paper', 'bn', 'adaptive'],
        help='Model architecture to use (default: paper)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable data augmentation'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['PASCAL', 'Physionet'],
        help='Dataset to use (default: None, prompts user)'
    )
    
    args = parser.parse_args()
    
    # Interactive dataset selection if not provided via arguments
    if args.dataset is None:
        print("\n" + "="*60)
        print("Select Dataset:")
        print("1. PASCAL")
        print("2. Physionet2016")
        print("="*60)
        
        while True:
            try:
                choice = input("Enter option (1 or 2): ").strip()
                if choice == '1':
                    args.dataset = 'PASCAL'
                    break
                elif choice == '2':
                    args.dataset = 'Physionet'
                    break
                else:
                    print("Invalid option. Please enter 1 or 2.")
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception:
                print("Invalid input.")
    
    print("\n" + "="*60)
    print(f"     HEART SOUND CLASSIFICATION - {args.dataset} Dataset")
    print("="*60)
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Augmentation: {not args.no_augment}")
    print("="*60 + "\n")
    
    # Train model
    # Note: train_with_kfold handles data loading internally
    best_model, fold_metrics = train_with_kfold(
        model_type=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k_folds=N_FOLDS
    )
    
    print("\n" + "="*60)
    print("                   TRAINING SUMMARY")
    print("="*60)
    print(f"  Accuracy:  {sum(fold_metrics['accuracy'])/len(fold_metrics['accuracy'])*100:.2f}%")
    print(f"  Precision: {sum(fold_metrics['precision'])/len(fold_metrics['precision'])*100:.2f}%")
    print(f"  Recall:    {sum(fold_metrics['recall'])/len(fold_metrics['recall'])*100:.2f}%")
    print(f"  F1 Score:  {sum(fold_metrics['f1'])/len(fold_metrics['f1'])*100:.2f}%")
    print("="*60)
    print("\nModel and results saved in 'models/' directory")
    

if __name__ == "__main__":
    main()
