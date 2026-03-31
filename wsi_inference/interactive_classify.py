"""
Interactive WSI Classification Interface
========================================
Easy way to pick an image and model, then classify and visualize.

Usage:
    python interactive_classify.py       # Interactive mode
    python interactive_classify.py --batch input_folder/  # Batch mode
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from .classify_wsi import WSIClassifier, get_available_models
import pandas as pd


def show_available_images(image_dir):
    """Show available images to classify."""
    image_dir = Path(image_dir)
    supported_formats = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    images = []
    
    for fmt in supported_formats:
        images.extend(sorted(image_dir.glob(f"*{fmt}")))
        images.extend(sorted(image_dir.glob(f"*{fmt.upper()}")))
    
    images = list(set(images))  # Remove duplicates
    images = sorted(images)
    
    return images


def pick_image_interactive(image_dir):
    """Interactive image selection."""
    images = show_available_images(image_dir)
    
    if not images:
        print(f"✗ No images found in {image_dir}")
        return None
    
    print(f"\nFound {len(images)} images:")
    print("=" * 80)
    for i, img_path in enumerate(images, 1):
        file_size_mb = img_path.stat().st_size / (1024**2)
        print(f"  {i:2d}. {img_path.name} ({file_size_mb:.1f} MB)")
    
    while True:
        try:
            choice = input(f"\nSelect image (1-{len(images)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(images):
                return images[idx]
            else:
                print(f"Please enter a number between 1 and {len(images)}")
        except ValueError:
            print("Please enter a valid number")


def pick_model_interactive(model_dir=None):
    """Interactive model selection.
    
    Parameters
    ----------
    model_dir : str, optional
        Path to models directory. If None, uses ./models relative to script.
    """
    models = get_available_models(models_dir=model_dir)
    
    if not models:
        if model_dir:
            print(f"✗ No trained models found in {model_dir}")
        else:
            print("✗ No trained models found in solution/models/")
        print("\nTrain a model first using:")
        print("  python ../train_domain_shift.py --scanner Hamamatsu_XR --augmentation standard")
        return None
    
    print(f"\nFound {len(models)} trained models:")
    print("=" * 80)
    for i, (name, path) in enumerate(models, 1):
        print(f"  {i:2d}. {name}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx][1]
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")


def interactive_mode():
    """Full interactive classification workflow."""
    print("\n" + "="*80)
    print("WSI PATCH CLASSIFICATION - INTERACTIVE MODE")
    print("="*80)
    
    # Hardcoded directories
    image_dir = Path(r"C:\DIPLOMSKI\MIDOG_Challenge\images")
    hardcoded_masks_dir = Path(r"C:\DIPLOMSKI\unet_env\code\solution\data\masks")
    
    print(f"\nImage directory: {image_dir}")
    print(f"Masks directory: {hardcoded_masks_dir}")
    
    if not image_dir.exists():
        print(f"✗ Image directory not found: {image_dir}")
        return
    
    if not hardcoded_masks_dir.exists():
        print(f"✗ Masks directory not found: {hardcoded_masks_dir}")
        return
    
    # Pick image
    image_path = pick_image_interactive(image_dir)
    if image_path is None:
        return
    
    print(f"\n✓ Selected: {image_path.name}")
    
    # Pick model
    model_path = pick_model_interactive()
    if model_path is None:
        return
    
    print(f"✓ Selected: {Path(model_path).stem}")
    
    # Classification parameters
    print(f"\nCustomization (press Enter for defaults):")
    
    stride_input = input("  Patch stride (default 64, smaller = more patches): ").strip()
    stride = int(stride_input) if stride_input else 64
    
    threshold_input = input("  Classification threshold (default 0.5): ").strip()
    try:
        threshold = float(threshold_input) if threshold_input else 0.5
    except ValueError:
        threshold = 0.5
    
    device_input = input("  Device (cuda/cpu, default cuda): ").strip().lower()
    device = device_input if device_input in ['cuda', 'cpu'] else 'cuda'
    
    output_dir = Path(__file__).parent / "output"
    
    # Run classification
    print(f"\n{'='*80}")
    print("Starting classification...")
    print(f"{'='*80}\n")
    
    classifier = WSIClassifier(model_path, device=device, stride=stride, masks_dir=hardcoded_masks_dir)
    heatmap, predictions, confidences, positions, image, mask = classifier.classify_wsi(
        image_path, 
        threshold=threshold
    )
    
    # Debug: Print confidence distribution
    print(f"\n🔍 CONFIDENCE STATISTICS:")
    print(f"  Min confidence: {confidences.min():.4f}")
    print(f"  Max confidence: {confidences.max():.4f}")
    print(f"  Mean confidence: {confidences.mean():.4f}")
    print(f"  Median confidence: {np.median(confidences):.4f}")
    print(f"  Std dev: {confidences.std():.4f}")
    print(f"  Threshold: {threshold}")
    print(f"  Predictions > threshold: {(confidences > threshold).sum()} / {len(confidences)}\n")
    
    # Visualize
    classifier.visualize_results(image, heatmap, output_dir, mask=mask)
    
    print("="*80)
    print("✓ CLASSIFICATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {image_path.stem}_original.png")
    print(f"  - {image_path.stem}_heatmap.png")
    print(f"  - {image_path.stem}_comparison.png")
    print(f"  - {image_path.stem}_detailed.png")
    print(f"  - {image_path.stem}_heatmap_raw.npy")


def batch_mode(image_folder=None, model_path=None, masks_dir=None):
    """Classify all images in a folder."""
    # Hardcode defaults
    if image_folder is None:
        image_folder = r"C:\DIPLOMSKI\MIDOG_Challenge\images"
    if masks_dir is None:
        masks_dir = r"C:\DIPLOMSKI\unet_env\code\solution\data\masks"
    """Classify all images in a folder."""
    image_folder = Path(image_folder)
    if not image_folder.exists():
        print(f"✗ Folder not found: {image_folder}")
        return
    
    # Get images
    images = show_available_images(image_folder)
    if not images:
        print(f"✗ No images found in {image_folder}")
        return
    
    # Get model
    if model_path is None:
        model_path = pick_model_interactive()
        if model_path is None:
            return
    
    print(f"\n{'='*80}")
    print(f"BATCH CLASSIFICATION: {len(images)} images")
    print(f"{'='*80}\n")
    
    output_dir = Path(__file__).parent / "output"
    classifier = WSIClassifier(model_path, masks_dir=masks_dir)
    
    results = []
    
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {img_path.name}")
        try:
            heatmap, predictions, confidences, positions, image, mask = classifier.classify_wsi(img_path)
            classifier.visualize_results(image, heatmap, output_dir, mask=mask)
            
            results.append({
                'image': img_path.name,
                'positive_patches': int(predictions.sum()),
                'total_patches': len(predictions),
                'positive_ratio': 100 * predictions.mean(),
                'mean_confidence': confidences.mean()
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'image': img_path.name,
                'error': str(e)
            })
    
    # Save summary
    df = pd.DataFrame(results)
    summary_path = output_dir / "batch_summary.csv"
    df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*80}")
    print("✓ BATCH CLASSIFICATION COMPLETE!")
    print("="*80)
    print(f"\nProcessed {len(images)} images")
    print(f"Summary: {summary_path}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive WSI classification")
    parser.add_argument('--batch', type=str, default=None,
                       help='Batch mode: classify all images in a folder')
    parser.add_argument('--model', type=str, default=None,
                       help='Model path (for batch mode)')
    parser.add_argument('--masks-dir', type=str, default=None,
                       help='Directory containing ground truth masks')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        models = get_available_models()
        if models:
            print("\nAvailable trained models:")
            print("=" * 80)
            for name, path in models:
                print(f"  {name}")
        else:
            print("No trained models found")
    elif args.batch:
        batch_mode(args.batch, args.model, args.masks_dir or r"C:\DIPLOMSKI\unet_env\code\solution\data\masks")
    else:
        interactive_mode()
