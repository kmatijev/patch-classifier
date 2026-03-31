"""
PHASE 1A: Load & Parse MIDOG Annotations
=========================================
Loads mitotic figure annotations from the MIDOG Challenge dataset.
Annotations are in COCO format (JSON) with bounding boxes and slide IDs.

INPUT:
------
MIDOG.json  - MIDOG Challenge annotations in COCO format
              Contains bounding boxes for all mitotic figures

OUTPUT:
-------
DataFrame with columns:
- file_name:    Original image filename
- image_id:     Unique image identifier
- width:        Image width in pixels
- height:       Image height in pixels
- box:          Bounding box [x1, y1, x2, y2] format
- cat:          Category ('mitotic figure')
- scanner:      Scanner model (Hamamatsu_XR, Hamamatsu_S360, Aperio_CS)

This DataFrame is used by subsequent processing steps.
"""

import json
import pandas as pd
from pathlib import Path


def load_midog_annotations(annotation_file_path, hamamatsu_xr_ids=None, hamamatsu_360_ids=None, 
                          aperio_ids=None, leica_ids=None):
    """
    Load MIDOG challenge annotations from COCO-formatted JSON.
    
    Parameters
    ----------
    annotation_file_path : str or Path
        Path to MIDOG.json annotation file
    
    hamamatsu_xr_ids : set, optional
        Image IDs acquired with Hamamatsu XR scanner
    
    hamamatsu_360_ids : set, optional
        Image IDs acquired with Hamamatsu S360 scanner
    
    aperio_ids : set, optional
        Image IDs acquired with Aperio CS scanner
    
    leica_ids : set, optional
        Image IDs acquired with Leica GT450 scanner
    
    Returns
    -------
    pd.DataFrame
        Annotations with columns: file_name, image_id, width, height, box, cat, scanner
    """
    
    # Initialize scanner ID sets if not provided
    if hamamatsu_xr_ids is None:
        hamamatsu_xr_ids = set()
    if hamamatsu_360_ids is None:
        hamamatsu_360_ids = set()
    if aperio_ids is None:
        aperio_ids = set()
    if leica_ids is None:
        leica_ids = set()
    
    # Default scanner type
    default_scanner = "Hamamatsu_XR"
    
    # Category mapping
    categories = {
        1: 'mitotic figure'      # Category 1: Positive mitotic figures
    }
    
    # Initialize list to store rows
    rows = []
    
    # Load MIDOG.json
    print(f"Loading annotations from: {annotation_file_path}")
    with open(annotation_file_path) as f:
        data = json.load(f)
    
    print(f"Total images in dataset: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}\n")
    
    # ====================================================================
    # STEP 1: Iterate through all images
    # ====================================================================
    for row in data["images"]:
        file_name = row["file_name"]
        image_id = row["id"]
        width = row["width"]
        height = row["height"]
        
        # ================================================================
        # STEP 2: Determine scanner type based on image ID
        # ================================================================
        scanner = default_scanner
        
        if image_id in hamamatsu_xr_ids:
            scanner = "Hamamatsu_XR"
        elif image_id in hamamatsu_360_ids:
            scanner = "Hamamatsu_S360"
        elif image_id in aperio_ids:
            scanner = "Aperio_CS"
        elif image_id in leica_ids:
            scanner = "Leica GT450"
        
        # ================================================================
        # STEP 3: Extract all annotations for this image
        # ================================================================
        for annotation in [anno for anno in data['annotations'] if anno["image_id"] == image_id]:
            category_id = annotation["category_id"]
            
            # Skip non-mitotic figure annotations
            if category_id not in categories:
                continue
            
            # Extract bounding box [x, y, width, height]
            box = annotation["bbox"]
            cat = categories[category_id]
            
            # ============================================================
            # STEP 4: Append row
            # ============================================================
            rows.append([file_name, image_id, width, height, box, cat, scanner])
    
    # ====================================================================
    # STEP 5: Create DataFrame
    # ====================================================================
    df = pd.DataFrame(rows, columns=[
        "file_name",
        "image_id",
        "width",
        "height",
        "box",
        "cat",
        "scanner"
    ])
    
    # Display statistics
    print(f"{'='*80}")
    print(f"ANNOTATIONS LOADED")
    print(f"{'='*80}")
    print(f"Total mitotic figure annotations: {len(df)}")
    print(f"Unique images: {df['image_id'].nunique()}")
    print(f"\nAnnotations per scanner:")
    print(df['scanner'].value_counts())
    print(f"{'='*80}\n")
    
    return df


# ====================================================================
# Usage Example
# ====================================================================
if __name__ == "__main__":
    # Define paths
    midog_folder = Path(r"C:\DIPLOMSKI\MIDOG_Challenge")
    annotation_file = midog_folder / "MIDOG.json"
    
    # Define scanner ID ranges
    # MIDOG dataset: 200 images total, 50 per scanner
    hamamatsu_xr_ids = set(range(0, 51))
    hamamatsu_360_ids = set(range(51, 101))
    aperio_ids = set(range(101, 151))
    leica_ids = set(range(151, 201))
    
    # Load annotations
    df = load_midog_annotations(
        annotation_file,
        hamamatsu_xr_ids=hamamatsu_xr_ids,
        hamamatsu_360_ids=hamamatsu_360_ids,
        aperio_ids=aperio_ids,
        leica_ids=leica_ids
    )
    
    print("First 5 annotations:")
    print(df.head())
