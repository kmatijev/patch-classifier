"""
WSI Patch Classification & Visualization Pipeline
=================================================
Load a large WSI image, classify patches, and visualize results.

WORKFLOW:
1. Load a WSI image
2. Split into 128×128 patches with stride
3. Classify each patch using trained model
4. Reconstruct heatmap (same size as original)
5. Visualize: Original image + Heatmap side by side
6. Save results with visualizations

OUTPUT:
-------
output/
  ├── [wsi_name]_original.png          (original image)
  ├── [wsi_name]_heatmap_raw.npy       (raw predictions - all patches)
  ├── [wsi_name]_heatmap.png           (heatmap visualization)
  ├── [wsi_name]_comparison.png        (side-by-side original + heatmap)
  └── [wsi_name]_detailed.png          (original + heatmap + patch grid overlay)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from argparse import ArgumentParser
import sys

# Add solution to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from patch_classifier_model import PatchClassifier


class WSIClassifier:
    """Classify large WSI images patch by patch."""
    
    def __init__(self, model_path, device="cuda", patch_size=128, stride=64, masks_dir=None):
        """
        Initialize classifier.
        
        Parameters
        ----------
        model_path : str or Path
            Path to trained model (.pth file)
        device : str
            Device to use: 'cuda' or 'cpu'
        patch_size : int
            Size of patches (default: 128×128)
        stride : int
            Stride between patches (default: 64 = 50% overlap)
        masks_dir : str or Path, optional
            Directory containing ground truth masks. If provided, masks will be loaded
            from here. If None, will search in solution/data/masks/ (default: None)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.stride = stride
        self.masks_dir = Path(masks_dir) if masks_dir else None
        
        
        # Load model
        self.model = PatchClassifier().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"✓ Loaded model from: {model_path}")
        print(f"✓ Using device: {self.device}")
        print(f"✓ Patch size: {patch_size}×{patch_size}")
        print(f"✓ Stride: {stride} pixels\n")
    
    def split_into_patches(self, image):
        """
        Split image into overlapping patches.
        
        Returns
        -------
        patches : list
            List of (patch, (y_start, x_start)) tuples
        height, width : int
            Original image dimensions
        """
        height, width = image.shape[:2]
        patches = []
        positions = []
        
        y = 0
        while y + self.patch_size <= height:
            x = 0
            while x + self.patch_size <= width:
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                positions.append((y, x))
                x += self.stride
            y += self.stride
        
        # Add edge patches if needed
        # Right edge
        if x + self.patch_size != width:
            x = width - self.patch_size
            for y in np.arange(0, height - self.patch_size + 1, self.stride):
                if (y, x) not in positions:
                    patch = image[y:y+self.patch_size, x:x+self.patch_size]
                    patches.append(patch)
                    positions.append((y, x))
        
        # Bottom edge
        if y + self.patch_size != height:
            y = height - self.patch_size
            for x in np.arange(0, width - self.patch_size + 1, self.stride):
                if (y, x) not in positions:
                    patch = image[y:y+self.patch_size, x:x+self.patch_size]
                    patches.append(patch)
                    positions.append((y, x))
        
        # Bottom-right corner
        if (height - self.patch_size, width - self.patch_size) not in positions:
            patch = image[height-self.patch_size:height, width-self.patch_size:width]
            patches.append(patch)
            positions.append((height - self.patch_size, width - self.patch_size))
        
        print(f"Split into {len(patches)} patches")
        return patches, positions, height, width
    
    def classify_patches(self, patches, threshold=0.5):
        """
        Classify all patches.
        
        Returns
        -------
        predictions : np.ndarray
            Binary predictions (0 or 1)
        confidences : np.ndarray
            Confidence scores [0, 1]
        """
        predictions = []
        confidences = []
        
        # ImageNet normalization (same as training)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        
        with torch.no_grad():
            for patch in tqdm(patches, desc="Classifying patches"):
                # Normalize to [0, 1]
                patch_norm = patch.astype(np.float32) / 255.0
                
                # Apply ImageNet normalization
                patch_norm = (patch_norm - imagenet_mean) / imagenet_std
                
                # Convert to tensor: HWC -> CHW
                patch_tensor = torch.from_numpy(patch_norm).permute(2, 0, 1).unsqueeze(0).float()
                patch_tensor = patch_tensor.to(self.device)
                
                # Classify
                output = self.model(patch_tensor)
                confidence = torch.sigmoid(output).item()
                prediction = 1 if confidence > threshold else 0
                
                predictions.append(prediction)
                confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def reconstruct_heatmap(self, predictions, confidences, positions, height, width):
        """
        Reconstruct full-size heatmap from patch predictions.
        
        Returns
        -------
        heatmap : np.ndarray
            Heatmap with same size as original image
        coverage : np.ndarray
            How many patches contributed to each pixel (for averaging)
        """
        heatmap = np.zeros((height, width), dtype=np.float32)
        coverage = np.zeros((height, width), dtype=np.float32)
        
        for (y, x), confidence in zip(positions, confidences):
            y_end = min(y + self.patch_size, height)
            x_end = min(x + self.patch_size, width)
            
            patch_height = y_end - y
            patch_width = x_end - x
            
            # Average confidence across patch region
            heatmap[y:y_end, x:x_end] += confidence
            coverage[y:y_end, x:x_end] += 1
        
        # Normalize by coverage (handle overlapping regions)
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = np.divide(heatmap, coverage, where=coverage > 0)
            heatmap = np.nan_to_num(heatmap, nan=0.0)
        
        return heatmap, coverage
    
    def find_mask_file(self, image_path):
        """
        Search for a corresponding ground truth mask file.
        
        Search order:
        1. If masks_dir specified: search in that directory
        2. Try {image_stem}_mask.{png,tiff,tif,jpg}
        3. Try {image_stem}.{png,tiff,tif,jpg}
        4. Try same directory as image with _mask suffix
        
        Returns
        -------
        mask_path : Path or None
            Path to mask file if found, else None
        """
        image_path = Path(image_path)
        image_stem = image_path.stem
        
        # If masks_dir was specified, use it
        if self.masks_dir:
            masks_dir = self.masks_dir
            print(f"  Searching for mask in specified directory: {masks_dir}")
        else:
            # Otherwise try solution/data/masks/ directory
            solution_root = Path(__file__).parent.parent
            masks_dir = solution_root / "data" / "masks"
            print(f"  Searching for mask in default directory: {masks_dir}")
        
        if masks_dir.exists():
            # Try with _mask suffix first (most common pattern)
            for ext in ['.tiff', '.tif', '.png', '.jpg']:
                mask_path = masks_dir / f"{image_stem}_mask{ext}"
                if mask_path.exists():
                    print(f"  ✓ Found: {mask_path}")
                    return mask_path
            
            # Try with same name
            for ext in ['.png', '.tiff', '.tif', '.jpg']:
                mask_path = masks_dir / f"{image_stem}{ext}"
                if mask_path.exists():
                    print(f"  ✓ Found: {mask_path}")
                    return mask_path
            
            print(f"  ✗ No mask found for {image_stem} (tried {image_stem}_mask.* and {image_stem}.*)")
        else:
            print(f"  ✗ Masks directory does not exist: {masks_dir}")
        
        # Try same directory as image with _mask suffix
        same_dir_mask = image_path.parent / f"{image_stem}_mask{image_path.suffix}"
        if same_dir_mask.exists():
            print(f"  ✓ Found in image directory: {same_dir_mask}")
            return same_dir_mask
        
        print(f"  ✗ Mask file not found for image: {image_path.name}")
        return None
    
    def load_mask(self, image_path):
        """
        Load and normalize ground truth mask.
        
        Parameters
        ----------
        image_path : str or Path
            Path to the WSI image
        
        Returns
        -------
        mask : np.ndarray or None
            Normalized mask [0, 1], or None if not found
        """
        mask_path = self.find_mask_file(image_path)
        
        if mask_path is None:
            return None
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  ⚠ Failed to load mask: {mask_path}")
            return None
        
        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        # Check if mask is empty
        num_white_pixels = np.sum(mask > 0.5)
        num_total_pixels = mask.size
        white_ratio = 100 * num_white_pixels / num_total_pixels
        
        print(f"  Mask stats: {white_ratio:.2f}% white pixels ({num_white_pixels} / {num_total_pixels})")
        
        return mask
    
    def _colorize_heatmap(self, heatmap):
        """
        Convert grayscale heatmap to colored visualization.
        
        Parameters
        ----------
        heatmap : np.ndarray
            Grayscale heatmap [0, 1]
        
        Returns
        -------
        heatmap_colored : np.ndarray
            RGB colored heatmap (H×W×3, values 0-255)
        """
        # Use matplotlib's RdYlBu_r colormap (Red=high, Blue=low)
        cmap = plt.get_cmap('RdYlBu_r')
        
        # Normalize heatmap to [0, 1] (already should be but ensure it)
        heatmap_normalized = np.clip(heatmap, 0, 1)
        
        # Apply colormap
        heatmap_colored = cmap(heatmap_normalized)  # Returns RGBA [0, 1]
        
        # Convert to RGB [0, 255]
        heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        return heatmap_rgb
    
    def classify_wsi(self, image_path, threshold=0.5):
        """
        Full pipeline: Load → Split → Classify → Reconstruct.
        
        Returns
        -------
        heatmap : np.ndarray
            Averaged heatmap [0, 1]
        predictions : np.ndarray
            Binary predictions per patch
        confidences : np.ndarray
            Confidence scores per patch
        positions : list
            (y, x) positions of patches
        image : np.ndarray
            Original image
        mask : np.ndarray or None
            Ground truth mask if found, else None
        """
        # Load image
        print(f"Loading image: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask if available
        mask = self.load_mask(image_path)
        if mask is not None:
            print(f"✓ Found corresponding mask")
        
        print(f"Image shape: {image.shape}\n")
        
        # Split into patches
        print("Splitting into patches...")
        patches, positions, height, width = self.split_into_patches(image)
        
        # Classify
        print("\nClassifying patches...")
        predictions, confidences = self.classify_patches(patches, threshold)
        
        # Reconstruct
        print("\nReconstructing heatmap...")
        heatmap, coverage = self.reconstruct_heatmap(predictions, confidences, positions, height, width)
        
        print(f"✓ Classification complete!")
        print(f"  Positive patches: {predictions.sum()} / {len(predictions)} ({100*predictions.mean():.1f}%)")
        print(f"  Mean confidence: {confidences.mean():.4f}")
        print(f"  Coverage min/max: {coverage.min():.0f} / {coverage.max():.0f}\n")
        
        return heatmap, predictions, confidences, positions, image, mask
    
    def visualize_results(self, image, heatmap, output_path, mask=None):
        """
        Create and save visualizations.
        
        Saves:
        1. Original image
        2. Mask (if available)
        3. Heatmap
        4. Side-by-side comparison (original + mask + heatmap)
        5. Detailed analysis with all views
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract name
        name = output_path.stem if output_path.is_file() else output_path.name
        if name == "output":
            name = "wsi_classification"
        
        height, width = image.shape[:2]
        
        # ================================================================
        # 1. Save original image
        # ================================================================
        original_path = output_path / f"{name}_original.png"
        cv2.imwrite(str(original_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # ================================================================
        # 2. Save mask if available
        # ================================================================
        if mask is not None:
            mask_viz_path = output_path / f"{name}_mask.png"
            mask_uint8 = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(mask_viz_path), mask_uint8)
        
        # ================================================================
        # 3. Save heatmap
        # ================================================================
        heatmap_path = output_path / f"{name}_heatmap.png"
        heatmap_colored = self._colorize_heatmap(heatmap)
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
        
        # ================================================================
        # 4. Side-by-side comparison (original + mask + heatmap)
        # ================================================================
        if mask is not None:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            
            axes[0].imshow(image)
            axes[0].set_title("Original WSI Image", fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title("Ground Truth Mask\n(White=Mitosis, Black=Background)", 
                             fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(heatmap_colored)
            axes[2].set_title("Model Prediction Heatmap\n(Red=Mitosis, Blue=Background)", 
                             fontsize=14, fontweight='bold')
            axes[2].axis('off')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            axes[0].imshow(image)
            axes[0].set_title("Original WSI Image", fontsize=16, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap_colored)
            axes[1].set_title("Patch Classification Heatmap\n(Red=Mitosis, Blue=Background)", 
                             fontsize=16, fontweight='bold')
            axes[1].axis('off')
        
        plt.tight_layout()
        comparison_path = output_path / f"{name}_comparison.png"
        plt.savefig(comparison_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # ================================================================
        # 5. Detailed analysis with all views (same size, with GT overlay)
        # ================================================================
        if mask is not None:
            fig, axes = plt.subplots(1, 4, figsize=(28, 7), constrained_layout=True)
            
            # Original
            axes[0].imshow(image)
            axes[0].set_title("Original WSI", fontsize=13, fontweight='bold')
            axes[0].axis('off')
            
            # Ground truth mask
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title("Ground Truth Mask", fontsize=13, fontweight='bold')
            axes[1].axis('off')
            
            # Heatmap
            axes[2].imshow(heatmap, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[2].set_title("Model Prediction Heatmap\n(Red=High, Blue=Low)", 
                         fontsize=13, fontweight='bold')
            axes[2].axis('off')
            
            # Overlay with ground truth mask
            axes[3].imshow(image, alpha=0.5)
            axes[3].imshow(heatmap, cmap='RdYlBu_r', vmin=0, vmax=1, alpha=0.5)
            
            # Add ground truth mask as bright green overlay
            mask_colored = np.zeros_like(image)
            mask_colored[:, :, 1] = (mask * 255).astype(np.uint8)  # Green channel - max intensity
            axes[3].imshow(mask_colored, alpha=0.6)
            
            axes[3].set_title("Prediction + Ground Truth Overlay\n(Bright Green=GT Mitosis)", 
                         fontsize=13, fontweight='bold')
            axes[3].axis('off')
        else:
            fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
            
            # Original
            axes[0].imshow(image)
            axes[0].set_title("Original WSI", fontsize=13, fontweight='bold')
            axes[0].axis('off')
            
            # Heatmap
            axes[1].imshow(heatmap, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[1].set_title("Model Prediction Heatmap\n(Red=High, Blue=Low)", 
                         fontsize=13, fontweight='bold')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(image, alpha=0.5)
            axes[2].imshow(heatmap, cmap='RdYlBu_r', vmin=0, vmax=1, alpha=0.5)
            axes[2].set_title("Prediction Overlay", fontsize=13, fontweight='bold')
            axes[2].axis('off')
        
        detailed_path = output_path / f"{name}_detailed.png"
        plt.savefig(detailed_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # ================================================================
        # 6. Save heatmap as numpy array (for further analysis)
        # ================================================================
        heatmap_npy = output_path / f"{name}_heatmap_raw.npy"
        np.save(heatmap_npy, heatmap)
        
        print("✓ Visualizations saved:")
        print(f"  {original_path}")
        if mask is not None:
            print(f"  {output_path / f'{name}_mask.png'}")
        print(f"  {heatmap_path}")
        print(f"  {comparison_path}")
        print(f"  {detailed_path}")
        print(f"  {heatmap_npy}\n")
        
        return {
            'original': original_path,
            'mask': output_path / f"{name}_mask.png" if mask is not None else None,
            'heatmap': heatmap_path,
            'comparison': comparison_path,
            'detailed': detailed_path,
            'heatmap_npy': heatmap_npy
        }


def get_available_models(models_dir=None):
    """List available trained models.
    
    Parameters
    ----------
    models_dir : str, optional
        Path to models directory. If None, uses ./models relative to script.
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"
    else:
        models_dir = Path(models_dir)
    
    if not models_dir.exists():
        return []
    
    models = sorted(models_dir.glob("patch_classifier_*.pth"))
    return [(m.stem.replace("patch_classifier_", ""), str(m)) for m in models]


def main():
    parser = ArgumentParser(description="Classify patches in WSI images")
    parser.add_argument('--image', type=str, required=True,
                       help='Path to WSI image to classify')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (or model name without path)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: solution/wsi_inference/output)')
    parser.add_argument('--masks-dir', type=str, default=None,
                       help='Directory containing ground truth masks (optional)')
    parser.add_argument('--stride', type=int, default=64,
                       help='Stride between patches (default: 64 = 50%% overlap)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available trained models')
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = get_available_models()
        if models:
            print("\nAvailable trained models:")
            print("=" * 80)
            for name, path in models:
                print(f"  {name}")
                print(f"    Path: {path}\n")
        else:
            print("No trained models found in solution/models/")
        return
    
    # Resolve paths
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Resolve model path
    model_path = Path(args.model)
    if not model_path.exists():
        # Try to find in models directory
        models_dir = Path(__file__).parent.parent / "models"
        alt_path = models_dir / f"patch_classifier_{args.model}.pth"
        if alt_path.exists():
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model not found: {args.model}")
    
    # Output directory
    output_dir = args.output_dir or (Path(__file__).parent / "output")
    output_dir = Path(output_dir)
    
    # Run classification
    print("="*80)
    print("WSI PATCH CLASSIFICATION & VISUALIZATION")
    print("="*80 + "\n")
    
    classifier = WSIClassifier(model_path, device=args.device, stride=args.stride, 
                              masks_dir=args.masks_dir)
    heatmap, predictions, confidences, positions, image, mask = classifier.classify_wsi(
        image_path, 
        threshold=args.threshold
    )
    
    # Visualize
    classifier.visualize_results(image, heatmap, output_dir, mask=mask)
    
    print("✓ Done! Check output directory for results.")


if __name__ == "__main__":
    main()
