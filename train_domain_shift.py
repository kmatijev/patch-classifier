"""
Training script for domain shift evaluation
Supports training on different scanners with different augmentations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from patch_classifier_model import PatchClassifier
from patch_classifier_dataset_augmented import PatchClassifierDatasetAugmented
from augmentation_strategies import get_augmentation, AugmentationInfo


def get_scanner_selection():
    """
    Interactive menu to select which scanner to train on
    Returns selected scanner
    """
    scanners = ["Hamamatsu_XR", "Hamamatsu_S360", "Aperio_CS"]
    
    print(f"\n{'='*100}")
    print("SELECT SCANNER TO TRAIN ON")
    print(f"{'='*100}")
    print("1. Hamamatsu_XR")
    print("2. Hamamatsu_S360")
    print("3. Aperio_CS")
    print(f"{'='*100}\n")
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            return scanners[0]
        elif choice == "2":
            return scanners[1]
        elif choice == "3":
            return scanners[2]
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def get_augmentation_selection():
    """
    Interactive menu to select which augmentation strategy to use
    Returns selected augmentation
    """
    augmentations = ["standard", "medium", "strong", "histology"]
    
    print(f"\n{'='*100}")
    print("SELECT AUGMENTATION STRATEGY")
    print(f"{'='*100}")
    print("1. Standard")
    print("2. Medium")
    print("3. Strong")
    print("4. Histology")
    print(f"{'='*100}\n")
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            return augmentations[0]
        elif choice == "2":
            return augmentations[1]
        elif choice == "3":
            return augmentations[2]
        elif choice == "4":
            return augmentations[3]
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")



def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.9).long() # povecaj threshold za manje FP
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'preds': np.array(all_preds),
        'labels': np.array(all_labels),
        'probs': np.array(all_probs)
    }


def train_classifier(data_root, scanner, augmentation, output_dir, device, batch_size=32, 
                     learning_rate=1e-3, num_epochs=50, num_workers=0):
    """
    Train patch classifier on specified scanner with specified augmentation
    
    Args:
        data_root: Path to multi-scanner patches root
        scanner: "Hamamatsu_XR", "Hamamatsu_S360", or "Aperio_CS"
        augmentation: "standard", "strong", or "histology"
        output_dir: Where to save model
        device: Torch device
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Number of epochs
    """
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Scanner:      {scanner}")
    print(f"Augmentation: {augmentation}")
    print(f"Data root:    {data_root / scanner}")
    print(f"Device:       {device}")
    print(f"Batch size:   {batch_size}")
    print(f"Learning rate:{learning_rate}")
    print(f"Epochs:       {num_epochs}")
    print(f"{'='*80}\n")
    
    # Get augmentation
    aug_transform = get_augmentation(augmentation)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = PatchClassifierDatasetAugmented(
        root=data_root,
        scanner=scanner,
        augmentation=aug_transform,
        split='train'
    )
    val_dataset = PatchClassifierDatasetAugmented(
        root=data_root,
        scanner=scanner,
        augmentation=None,
        split='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Model
    print("Initializing model...")
    model = PatchClassifier().to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # Loss and optimizer
    # pos_weight=2.0 makes model more conservative (reduces FP), effectively raising threshold
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0], device=device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # Training loop
    print("Starting training...\n")
    best_f1 = 0.0
    best_model_state = None
    training_start_time = time.time()
    epoch_times = []
    
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        epoch_start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            marker = " ← BEST"
        else:
            marker = ""
        
        # Calculate ETA
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_str = f"{int(eta_seconds//60):d}m {int(eta_seconds%60):d}s" if eta_seconds > 0 else "Done"
        epoch_str = f"{int(epoch_duration//60):d}m {int(epoch_duration%60):d}s"
        
        tqdm.write(f"Epoch {epoch+1:2d}/{num_epochs} | "
                   f"Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                   f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f} | "
                   f"Time: {epoch_str} | ETA: {eta_str}{marker}")
        
        scheduler.step()
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    avg_epoch_time = np.mean(epoch_times)
    
    # Save best model
    model.load_state_dict(best_model_state)
    model_name = f"patch_classifier_{scanner.replace(' ', '_')}_{augmentation}.pth"
    model_path = output_dir / model_name
    torch.save(model.state_dict(), model_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    total_mins = int(total_training_time // 60)
    total_secs = int(total_training_time % 60)
    avg_mins = int(avg_epoch_time // 60)
    avg_secs = int(avg_epoch_time % 60)
    print(f"Total time:     {total_mins}m {total_secs}s")
    print(f"Avg per epoch:  {avg_mins}m {avg_secs}s")
    print(f"Best F1:        {best_f1:.4f}")
    print(f"Model saved:    {model_path}\n")
    
    return model_path


def main():
    # Default paths relative to solution folder
    solution_dir = Path(__file__).parent
    default_data_root = str(solution_dir / "data" / "patches" / "multi_scanner")
    
    parser = argparse.ArgumentParser(description='Train patch classifier with domain shift')
    parser.add_argument('--data-root', type=str, default=default_data_root,
                       help='Path to multi-scanner patches')
    parser.add_argument('--scanner', type=str, default=None,
                       choices=['Hamamatsu_XR', 'Hamamatsu_S360', 'Aperio_CS'],
                       help='Scanner to train on (if not provided, will prompt)')
    parser.add_argument('--augmentation', type=str, default=None,
                       choices=['standard', 'medium', 'strong', 'histology'],
                       help='Augmentation strategy (if not provided, will prompt)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Where to save trained models (auto-detected from data-root if not provided)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0,
                       help='Number of workers for data loading (default: 0). Set to 4-8 for faster GPU utilization')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    # Auto-detect patch size from data-root and set output directory
    if args.output_dir is None:
        data_root_path = Path(args.data_root)
        # Check if data_root contains 'data_XXX' pattern (e.g., data_256, data_384)
        if 'data_' in str(data_root_path):
            # Extract the patch size from path like "data_256" or "data_384"
            for part in data_root_path.parts:
                if part.startswith('data_'):
                    patch_size = part.replace('data_', '')
                    args.output_dir = str(solution_dir / f"models_{patch_size}")
                    break
        
        # Default to models/ if no patch size detected
        if args.output_dir is None:
            args.output_dir = str(solution_dir / "models")
    
    # If scanner not provided via CLI, prompt user
    if args.scanner is None:
        args.scanner = get_scanner_selection()
    
    # If augmentation not provided via CLI, prompt user
    if args.augmentation is None:
        args.augmentation = get_augmentation_selection()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_classifier(
        data_root=args.data_root,
        scanner=args.scanner,
        augmentation=args.augmentation,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
