"""
Train all models for domain shift evaluation
3 scanners × 4 augmentation strategies = 12 models
"""

import subprocess
import sys
from pathlib import Path


def train_all_models(data_root=r"C:\DIPLOMSKI\unet_env\code\solution\data", 
                     model_dir=r"C:\DIPLOMSKI\unet_env\code\solution\models",
                     batch_size=32, lr=1e-3, epochs=50):
    """
    Train all 12 combinations of (scanner × augmentation)
    """
    
    scanners = ["Hamamatsu_XR", "Hamamatsu_S360", "Aperio_CS"]
    augmentations = ["standard", "medium", "strong", "histology"]
    
    print(f"\n{'='*100}")
    print("TRAINING ALL 12 MODELS FOR DOMAIN SHIFT ANALYSIS")
    print(f"{'='*100}\n")
    print(f"Data root:   {data_root}")
    print(f"Model dir:   {model_dir}")
    print(f"Batch size:  {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs:      {epochs}")
    print(f"Total models: {len(scanners)} × {len(augmentations)} = 12\n")
    
    model_count = 0
    failed = []
    
    for scanner in scanners:
        for aug in augmentations:
            model_count += 1
            print(f"\n[{model_count}/12] Training {scanner} + {aug}...")
            print("-" * 100)
            
            cmd = [
                sys.executable, "-m", "train_domain_shift",
                "--data-root", data_root,
                "--scanner", scanner,
                "--augmentation", aug,
                "--output-dir", model_dir,
                "--batch-size", str(batch_size),
                "--lr", str(lr),
                "--epochs", str(epochs)
            ]
            
            try:
                result = subprocess.run(cmd, check=True)
                if result.returncode != 0:
                    failed.append(f"{scanner} + {aug}")
            except subprocess.CalledProcessError:
                failed.append(f"{scanner} + {aug}")
                print(f"Training failed for {scanner} + {aug}")
    
    print(f"\n\n{'='*100}")
    print("TRAINING SUMMARY")
    print(f"{'='*100}")
    print(f"Successfully trained: {12 - len(failed)} models")
    if failed:
        print(f"Failed: {len(failed)} models")
        for f in failed:
            print(f"   - {f}")
    else:
        print(f"All 12 models trained successfully!")
    print(f"{'='*100}\n")
    
    return len(failed) == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all 12 models for domain shift analysis')
    parser.add_argument('--data-root', default=r"C:\DIPLOMSKI\unet_env\code\patch_classifier\patches\multi_scanner")
    parser.add_argument('--model-dir', default=r"./models")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    success = train_all_models(
        data_root=args.data_root,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs
    )
    
    if not success:
        sys.exit(1)
