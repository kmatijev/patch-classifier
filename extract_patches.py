"""
PHASE 1C: Extract Multi-Scanner Patches
========================================
Extracts 128×128 patches from WSI and masks.

PROCESS:
--------
For each WSI (per scanner):
1. Extract patches around mitosis coordinates (positive patches)
   - 7 overlapping patches per mitosis with different offsets
2. Extract background patches (negative patches)
   - Random patches from background regions
3. Balance dataset (~45-50% positive)
4. Auto-split: 70% train, 15% val, 15% test per scanner

OUTPUT:
-------
Directory structure:
patch_classifier/patches/multi_scanner/
├── Hamamatsu_XR/
│   ├── train/
│   │   ├── images/    ~2800 patches (128×128 PNG)
│   │   └── masks/     ~2800 binary masks
│   ├── val/
│   │   ├── images/    ~600 patches
│   │   └── masks/     ~600 masks
│   └── test/
│       ├── images/    ~600 patches
│       └── masks/     ~600 masks
├── Hamamatsu_S360/
│   ├── train/
│   ├── val/
│   └── test/
└── Aperio_CS/
    ├── train/
    ├── val/
    └── test/

NOTES:
------
- ~8,000+ patches per scanner total
- Patches are 128×128 RGB images (PNG format)
- Masks are binary images (0=background, 1=mitosis)
- Dataset is automatically stratified by scanner
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def extract_multi_scanner_patches(df, images_dir, masks_dir, output_dir, 
                                  patch_size=128, positive_ratio=0.5, 
                                  exclude_scanner="Leica GT450",
                                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Extract multi-scanner patches with automatic train/val/test split.
    
    Parameters
    ----------
    df : pd.DataFrame
        Annotations DataFrame with mask_path column
    
    images_dir : str or Path
        Path to original TIFF images
    
    masks_dir : str or Path
        Path to mask TIFF files
    
    output_dir : str or Path
        Output directory for patches
    
    patch_size : int
        Patch size in pixels (default: 128)
    
    positive_ratio : float
        Target ratio of positive patches (default: 0.5 = 50%)
    
    exclude_scanner : str, optional
        Scanner to exclude from processing
    
    train_ratio : float
        Ratio of data for training (default: 0.7 = 70%)
    
    val_ratio : float
        Ratio of data for validation (default: 0.15 = 15%)
    
    test_ratio : float
        Ratio of data for testing (default: 0.15 = 15%)
    
    Returns
    -------
    dict
        Statistics about extracted patches per scanner and split
    """
    
    # Convert paths to Path objects
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    
    print(f"\n{'='*80}")
    print(f"PHASE 1C: EXTRACT MULTI-SCANNER PATCHES")
    print(f"{'='*80}")
    print(f"Patch size: {patch_size}×{patch_size}")
    print(f"Target positive ratio: {positive_ratio*100:.0f}%")
    print(f"Train/Val/Test split: {train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {test_ratio*100:.0f}%")
    print(f"Output directory: {output_dir}\n")
    
    # Filter by scanner
    if exclude_scanner:
        print(f"Excluding scanner: {exclude_scanner}")
        df = df[df['scanner'] != exclude_scanner]
        print(f"Annotations after filter: {len(df)}\n")
    
    # Get unique scanners
    scanners = sorted(df['scanner'].unique())
    print(f"Processing {len(scanners)} scanners: {', '.join(scanners)}\n")
    
    all_statistics = {}
    
    # ====================================================================
    # STEP 1: Process each scanner separately
    # ====================================================================
    for scanner in scanners:
        print(f"\n{'='*80}")
        print(f"SCANNER: {scanner}")
        print(f"{'='*80}\n")
        
        # Filter by scanner
        scanner_df = df[df['scanner'] == scanner].copy()
        unique_images = scanner_df[['file_name', 'image_id']].drop_duplicates()
        
        print(f"Processing {len(unique_images)} images...\n")
        
        all_patches = []
        
        # ================================================================
        # STEP 2: Extract patches from each image
        # ================================================================
        for img_idx, (_, image_row) in enumerate(unique_images.iterrows()):
            image_id = image_row['image_id']
            
            # Load image
            image_path = images_dir / image_row['file_name']
            if not image_path.exists():
                # Try other extensions
                stem = image_path.stem
                parent = image_path.parent
                for ext in ['.tiff', '.tif', '.jpg', '.jpeg', '.png']:
                    alt_path = parent / (stem + ext)
                    if alt_path.exists():
                        image_path = alt_path
                        break
            
            if not image_path.exists():
                print(f"  ⚠ Image not found: {image_path}")
                continue
            
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  ⚠ Could not read image: {image_path}")
                continue
            
            height, width = img.shape[:2]
            
            # Load mask
            mask_filename = Path(image_row['file_name']).stem + '_mask.tiff'
            mask_path = masks_dir / mask_filename
            
            if not mask_path.exists():
                print(f"  ⚠ Mask not found: {mask_path}")
                continue
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  ⚠ Could not read mask: {mask_path}")
                continue
            
            # ============================================================
            # STEP 3: Extract positive patches (around mitosis)
            # ============================================================
            positive_patches = []
            
            # Find mitosis regions in mask
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # For each mitosis, extract multiple patches
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Extract patches at different offsets around mitosis
                offsets = [
                    (0, 0),           # center
                    (-30, 0),         # left  (absolute pixels)
                    (30, 0),          # right
                    (0, -30),         # top
                    (0, 30),          # bottom
                    (-30, -30),       # top-left
                    (30, 30),         # bottom-right
                ]
                
                for offset_x, offset_y in offsets:
                    patch_center_x = center_x + offset_x
                    patch_center_y = center_y + offset_y
                    
                    # Extract patch centered on offset point
                    y_start = max(0, patch_center_y - patch_size // 2)
                    x_start = max(0, patch_center_x - patch_size // 2)
                    y_end = min(height, y_start + patch_size)
                    x_end = min(width, x_start + patch_size)
                    
                    # Adjust if patch goes out of bounds
                    if y_end - y_start < patch_size:
                        y_start = max(0, y_end - patch_size)
                    if x_end - x_start < patch_size:
                        x_start = max(0, x_end - patch_size)
                    
                    y_end = min(height, y_start + patch_size)
                    x_end = min(width, x_start + patch_size)
                    
                    # Extract patch if within bounds
                    if y_end - y_start == patch_size and x_end - x_start == patch_size:
                        patch = img[y_start:y_end, x_start:x_end]
                        patch_mask = mask[y_start:y_end, x_start:x_end]
                        
                        patch_info = {
                            'patch_data': patch,
                            'mask_data': patch_mask,
                            'label': 'positive',
                            'image_id': image_id,
                            'offset_idx': len(positive_patches)
                        }
                        positive_patches.append(patch_info)
            
            # ============================================================
            # STEP 4: Extract negative patches (background regions)
            # ============================================================
            negative_patches = []
            
            # Target number of negative patches
            if len(positive_patches) > 0:
                target_negative = max(1, int(len(positive_patches) * (1 - positive_ratio) / positive_ratio))
            else:
                target_negative = 10
            
            attempts = 0
            max_attempts = 5000
            
            while len(negative_patches) < target_negative and attempts < max_attempts:
                if height > patch_size and width > patch_size:
                    y_start = np.random.randint(0, height - patch_size + 1)
                    x_start = np.random.randint(0, width - patch_size + 1)
                else:
                    break
                
                y_end = y_start + patch_size
                x_end = x_start + patch_size
                
                # Check if patch is mostly background (< 5% mitotic coverage)
                patch_mask = mask[y_start:y_end, x_start:x_end]
                coverage = np.sum(patch_mask > 0) / (patch_size * patch_size)
                
                if coverage < 0.05:
                    patch = img[y_start:y_end, x_start:x_end]
                    
                    patch_info = {
                        'patch_data': patch,
                        'mask_data': patch_mask,
                        'label': 'negative',
                        'image_id': image_id,
                        'offset_idx': len(negative_patches)
                    }
                    negative_patches.append(patch_info)
                
                attempts += 1
            
            # Store patches
            all_patches.extend(positive_patches)
            all_patches.extend(negative_patches)
            
            if (img_idx + 1) % 5 == 0:
                print(f"  Processed {img_idx + 1}/{len(unique_images)} images")
                print(f"    Current: {len(positive_patches)} positive + {len(negative_patches)} negative")
        
        # ================================================================
        # STEP 5: Statistics and split
        # ================================================================
        n_positive = sum(1 for p in all_patches if p['label'] == 'positive')
        n_negative = sum(1 for p in all_patches if p['label'] == 'negative')
        n_total = len(all_patches)
        actual_ratio = n_positive / n_total if n_total > 0 else 0
        
        print(f"\nExtracted patches:")
        print(f"  Positive: {n_positive}")
        print(f"  Negative: {n_negative}")
        print(f"  Total: {n_total}")
        print(f"  Actual ratio: {actual_ratio*100:.1f}% positive\n")
        
        # ================================================================
        # STEP 6: Train/Val/Test split
        # ================================================================
        # Create train/test split first (80/20)
        train_patches, test_patches = train_test_split(
            all_patches, 
            test_size=(1 - train_ratio), 
            random_state=42,
            stratify=[p['label'] for p in all_patches]
        )
        
        # Further split test into val/test (50/50)
        val_patches, test_patches = train_test_split(
            test_patches,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=42,
            stratify=[p['label'] for p in test_patches]
        )
        
        splits = {
            'train': train_patches,
            'val': val_patches,
            'test': test_patches
        }
        
        print(f"Split distribution:")
        for split_name, split_patches in splits.items():
            n_pos = sum(1 for p in split_patches if p['label'] == 'positive')
            n_neg = sum(1 for p in split_patches if p['label'] == 'negative')
            pct = 100 * len(split_patches) / n_total if n_total > 0 else 0
            print(f"  {split_name:5s}: {len(split_patches):4d} patches ({pct:.1f}%) | "
                  f"{n_pos:3d} positive, {n_neg:3d} negative")
        
        # ================================================================
        # STEP 7: Save patches to disk
        # ================================================================
        scanner_folder = scanner.replace(" ", "_")
        
        for split_name, split_patches in splits.items():
            # Create directory structure
            split_images_dir = output_dir / scanner_folder / split_name / "images"
            split_masks_dir = output_dir / scanner_folder / split_name / "masks"
            
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_masks_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each patch
            for patch_idx, patch_info in enumerate(split_patches):
                patch_file = f"{patch_idx:06d}_{patch_info['label']}.png"
                
                image_path = split_images_dir / patch_file
                mask_path = split_masks_dir / patch_file
                
                cv2.imwrite(str(image_path), patch_info['patch_data'])
                cv2.imwrite(str(mask_path), patch_info['mask_data'])
        
        print(f"\nPatches saved to: {output_dir / scanner_folder}\n")
        
        # Store statistics
        all_statistics[scanner] = {
            'total_patches': n_total,
            'positive_patches': n_positive,
            'negative_patches': n_negative,
            'positive_ratio': actual_ratio,
            'train': len(train_patches),
            'val': len(val_patches),
            'test': len(test_patches)
        }
    
    # ====================================================================
    # STEP 8: Summary
    # ====================================================================
    print(f"\n{'='*80}")
    print(f"PATCH EXTRACTION COMPLETE")
    print(f"{'='*80}\n")
    
    for scanner, stats in all_statistics.items():
        print(f"{scanner}:")
        print(f"  Total: {stats['total_patches']} patches")
        print(f"  Positive: {stats['positive_patches']} ({stats['positive_ratio']*100:.1f}%)")
        print(f"  Negative: {stats['negative_patches']}")
        print(f"  Train/Val/Test: {stats['train']} / {stats['val']} / {stats['test']}")
        print()
    
    print(f"Output structure: {output_dir}/")
    for scanner in scanners:
        scanner_folder = scanner.replace(" ", "_")
        print(f"  {scanner_folder}/")
        for split in ['train', 'val', 'test']:
            print(f"    {split}/")
            print(f"      ├── images/")
            print(f"      └── masks/")
    
    print(f"\n✓ Patches ready for training!\n")
    
    return all_statistics


# ====================================================================
# Usage Example
# ====================================================================
if __name__ == "__main__":
    import argparse
    from load_midog_annotations import load_midog_annotations
    from generate_masks import generate_masks_from_annotations
    
    parser = argparse.ArgumentParser(description='Extract patches from WSI')
    parser.add_argument('--patch-size', type=int, default=128,
                       help='Patch size in pixels (default: 128)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: data/ for 128px, data_256/ for 256px, etc.)')
    parser.add_argument('--masks-dir', type=str, default=None,
                       help='Masks directory (default: data/masks for 128px, data_256/masks for 256px, etc.)')
    parser.add_argument('--positive-ratio', type=float, default=0.5,
                       help='Target positive patch ratio (default: 0.5 = 50%%)')
    args = parser.parse_args()
    
    # Define paths
    midog_folder = Path(r"C:\DIPLOMSKI\MIDOG_Challenge")
    annotation_file = midog_folder / "MIDOG.json"
    images_dir = midog_folder / "images"
    
    # Auto-select output directory based on patch size if not provided
    if args.output_dir is None:
        if args.patch_size == 128:
            output_dir = Path(__file__).parent / "data" / "patches" / "multi_scanner"
        else:
            output_dir = Path(__file__).parent / f"data_{args.patch_size}" / "patches" / "multi_scanner"
    else:
        output_dir = Path(args.output_dir)
    
    # Auto-select masks directory based on patch size if not provided
    if args.masks_dir is None:
        if args.patch_size == 128:
            masks_dir = Path(__file__).parent / "data" / "masks"
        else:
            masks_dir = Path(__file__).parent / f"data_{args.patch_size}" / "masks"
    else:
        masks_dir = Path(args.masks_dir)
    
    # Define scanner ID ranges
    hamamatsu_xr_ids = set(range(0, 51))
    hamamatsu_360_ids = set(range(51, 101))
    aperio_ids = set(range(101, 151))
    leica_ids = set(range(151, 201))
    
    # Phase 1A: Load annotations
    print("Phase 1A: Loading annotations...")
    df = load_midog_annotations(
        annotation_file,
        hamamatsu_xr_ids=hamamatsu_xr_ids,
        hamamatsu_360_ids=hamamatsu_360_ids,
        aperio_ids=aperio_ids,
        leica_ids=leica_ids
    )
    
    # Phase 1B: Generate masks
    print("\nPhase 1B: Generating masks...")
    df = generate_masks_from_annotations(
        df,
        images_dir=images_dir,
        output_dir=masks_dir,
        exclude_scanner="Leica GT450"
    )
    
    # Phase 1C: Extract patches with specified size
    print(f"\nPhase 1C: Extracting {args.patch_size}x{args.patch_size} patches...")
    statistics = extract_multi_scanner_patches(
        df,
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        patch_size=args.patch_size,
        positive_ratio=args.positive_ratio,
        exclude_scanner="Leica GT450"
    )
    
    print(f"✓ Data extraction complete! Patch size: {args.patch_size}px")
