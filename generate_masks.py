"""
PHASE 1B: Generate Binary Masks
================================
Creates binary masks from annotations where:
- Mitotic figures = 255 (white)
- Background = 0 (black)

One mask file per WSI, same dimensions as original image.

INPUT:
------
- Annotation DataFrame (from phase1_load_midog_annotations.py)
- Original TIFF files (from MIDOG dataset)

OUTPUT:
-------
- Binary mask TIFF files (one per image)
- Updated DataFrame with mask_path column
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def generate_masks_from_annotations(df, images_dir, output_dir, exclude_scanner=None):
    """
    Generate binary masks from annotation dataframe.
    
    For each image, creates a mask where mitotic figure bounding boxes
    are drawn as white regions (255) on black background (0).
    
    Parameters
    ----------
    df : pd.DataFrame
        Annotations DataFrame from load_midog_annotations()
    
    images_dir : str or Path
        Path to directory containing original TIFF images
    
    output_dir : str or Path
        Path where mask files will be saved
    
    exclude_scanner : str, optional
        Scanner to exclude from processing (e.g., 'Leica GT450')
    
    Returns
    -------
    pd.DataFrame
        Updated DataFrame with 'mask_path' column
    """
    
    # Convert paths to Path objects
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"PHASE 1B: MASK GENERATION")
    print(f"{'='*80}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Filter by scanner if needed
    if exclude_scanner:
        print(f"Excluding scanner: {exclude_scanner}")
        df = df[df['scanner'] != exclude_scanner]
        print(f"Annotations after filter: {len(df)}\n")
    
    # Get unique images
    unique_images = df[['file_name', 'image_id', 'width', 'height']].drop_duplicates()
    print(f"Generating masks for {len(unique_images)} unique images\n")
    
    mask_paths = []
    
    # ====================================================================
    # STEP 1: Iterate through each unique image
    # ====================================================================
    for idx, (_, row) in enumerate(unique_images.iterrows()):
        file_name = row['file_name']
        image_id = row['image_id']
        width = int(row['width'])
        height = int(row['height'])
        
        # ================================================================
        # STEP 2: Load the original image
        # ================================================================
        image_path = images_dir / file_name
        
        # Try different extensions if file not found
        if not image_path.exists():
            stem = image_path.stem
            parent = image_path.parent
            for ext in ['.tiff', '.tif', '.jpg', '.jpeg', '.png']:
                alt_path = parent / (stem + ext)
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        if not image_path.exists():
            print(f"  ⚠ WARNING: Image not found: {image_path}")
            continue
        
        # Load image to verify dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  ⚠ WARNING: Could not read image: {image_path}")
            continue
        
        # ================================================================
        # STEP 3: Create blank mask (black background)
        # ================================================================
        # Single channel, 8-bit unsigned integer
        # 0 = background, 255 = mitotic figure
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # ================================================================
        # STEP 4: Draw bounding boxes on mask
        # ================================================================
        # Get all annotations for this image
        image_annotations = df[df['image_id'] == image_id]
        
        for _, anno_row in image_annotations.iterrows():
            box = anno_row['box']  # [x, y, x_end, y_end] format
            
            # Extract box coordinates
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            
            # Draw filled rectangle on mask (white = 255)
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)
        
        # ================================================================
        # STEP 5: Save mask file
        # ================================================================
        mask_filename = Path(file_name).stem + '_mask.tiff'
        mask_path = output_dir / mask_filename
        
        cv2.imwrite(str(mask_path), mask)
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(unique_images)} images")
        
        mask_paths.append((image_id, str(mask_path)))
    
    # ====================================================================
    # STEP 6: Add mask paths to DataFrame
    # ====================================================================
    mask_path_dict = {image_id: str(mask_path) for image_id, mask_path in mask_paths}
    df['mask_path'] = df['image_id'].map(mask_path_dict)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"MASK GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total masks created: {len(mask_paths)}")
    print(f"Masks saved to: {output_dir}\n")
    
    return df


# ====================================================================
# Usage Example
# ====================================================================
if __name__ == "__main__":
    from load_midog_annotations import load_midog_annotations
    
    # Define paths
    midog_folder = Path(r"C:\DIPLOMSKI\MIDOG_Challenge")
    annotation_file = midog_folder / "MIDOG.json"
    images_dir = midog_folder / "images"
    output_dir = Path(__file__).parent / "data" / "masks"
    
    # Define scanner ID ranges
    hamamatsu_xr_ids = set(range(0, 51))
    hamamatsu_360_ids = set(range(51, 101))
    aperio_ids = set(range(101, 151))
    leica_ids = set(range(151, 201))
    
    # Step 1A: Load annotations
    print("Step 1A: Loading annotations...")
    df = load_midog_annotations(
        annotation_file,
        hamamatsu_xr_ids=hamamatsu_xr_ids,
        hamamatsu_360_ids=hamamatsu_360_ids,
        aperio_ids=aperio_ids,
        leica_ids=leica_ids
    )
    
    # Step 1B: Generate masks
    print("\nStep 1B: Generating masks...")
    df = generate_masks_from_annotations(
        df,
        images_dir=images_dir,
        output_dir=output_dir,
        exclude_scanner="Leica GT450"
    )
    
    # Display sample
    print(f"DataFrame with mask paths:")
    print(df[['file_name', 'image_id', 'mask_path', 'scanner']].head(10))
