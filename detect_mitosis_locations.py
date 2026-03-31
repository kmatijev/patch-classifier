"""
Mitosis Detection from WSI Heatmap
==================================
Processes model prediction heatmap to:
1. Detect mitosis locations (peaks) using blur + threshold + NMS
2. Match predictions with ground truth annotations
3. Calculate detection metrics (precision, recall, F1)
4. Visualize results

Usage:
    python detect_mitosis_locations.py
    - Interactive mode: select image and model
    - Then set threshold and evaluate
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys

sys.path.insert(0, str(Path(__file__).parent))

from wsi_inference.classify_wsi import WSIClassifier, get_available_models
from wsi_inference.interactive_classify import show_available_images, pick_image_interactive, pick_model_interactive


class MitosisDetector:
    """Detect and localize mitosis from WSI heatmap."""
    
    def __init__(self, classifier):
        """
        Parameters
        ----------
        classifier : WSIClassifier
            Trained WSI classifier
        """
        self.classifier = classifier
    
    def detect_peaks(self, heatmap, threshold=0.5, min_distance=30, blur_sigma=2.0, debug=True):
        """
        Detect mitosis peaks from heatmap using blur + NMS + confidence filtering.
        
        Parameters
        ----------
        heatmap : np.ndarray
            Model confidence heatmap [H, W] with values [0, 1]
        threshold : float
            Confidence threshold for peak detection (default: 0.5)
        min_distance : int
            Minimum distance between peaks in pixels (default: 30). Mitosis are typically
            separated by at least 30-50 pixels, so smaller values create false positives.
        blur_sigma : float
            Gaussian blur sigma (default: 2.0). Stronger blur merges nearby patch predictions.
        debug : bool
            Print intermediate statistics (default: True)
        
        Returns
        -------
        peaks : list of tuples
            List of (y, x) peak coordinates
        confidences : list of floats
            Confidence value at each peak
        """
        # Normalize heatmap
        heatmap_norm = np.clip(heatmap, 0, 1)
        
        # Apply strong Gaussian blur to smooth and connect nearby peaks
        heatmap_blurred = cv2.GaussianBlur(
            (heatmap_norm * 255).astype(np.uint8),
            ksize=(int(blur_sigma * 2) * 2 + 1, int(blur_sigma * 2) * 2 + 1),
            sigmaX=blur_sigma,
            sigmaY=blur_sigma
        ).astype(np.float32) / 255.0
        
        if debug:
            print(f"  After blur - Min: {heatmap_blurred.min():.4f}, Max: {heatmap_blurred.max():.4f}, Mean: {heatmap_blurred.mean():.4f}")
        
        # Apply threshold first to filter candidates down (avoid checking millions of pixels)
        threshold_mask = heatmap_blurred > threshold
        if not np.any(threshold_mask):
            return [], []
        
        # Non-maximum suppression on threshold-filtered region only
        footprint_size = min_distance * 2 + 1
        heatmap_local_max = maximum_filter(
            heatmap_blurred,
            size=(footprint_size, footprint_size)
        ) == heatmap_blurred
        
        # Keep only peaks above threshold
        heatmap_local_max[~threshold_mask] = False
        
        # Extract peaks
        peaks_y, peaks_x = np.where(heatmap_local_max)
        peaks = [(y, x) for y, x in zip(peaks_y, peaks_x)]
        confidences = [heatmap_blurred[y, x] for y, x in peaks]
        
        if debug:
            print(f"  Final peaks above threshold {threshold:.2f}: {len(peaks)}")
            if len(confidences) > 0:
                print(f"  Confidence stats: min={np.min(confidences):.4f}, max={np.max(confidences):.4f}, mean={np.mean(confidences):.4f}")
        
        return peaks, confidences
    
    def detect_peaks_greedy(self, heatmap, threshold=0.5, min_distance=30, blur_sigma=2.0, debug=True):
        """
        Detect mitosis peaks using greedy algorithm (iteratively pick highest peaks, suppress neighbors).
        Fast and avoids the maximum_filter noise problem.
        
        Parameters
        ----------
        heatmap : np.ndarray
            Model confidence heatmap [H, W] with values [0, 1]
        threshold : float
            Confidence threshold for peak detection (default: 0.5)
        min_distance : int
            Minimum distance between peaks in pixels (default: 30)
        blur_sigma : float
            Gaussian blur sigma (default: 2.0)
        debug : bool
            Print intermediate statistics (default: True)
        
        Returns
        -------
        peaks : list of tuples
            List of (y, x) peak coordinates
        confidences : list of floats
            Confidence value at each peak
        """
        # Normalize and blur
        heatmap_norm = np.clip(heatmap, 0, 1)
        heatmap_blurred = cv2.GaussianBlur(
            (heatmap_norm * 255).astype(np.uint8),
            ksize=(int(blur_sigma * 2) * 2 + 1, int(blur_sigma * 2) * 2 + 1),
            sigmaX=blur_sigma,
            sigmaY=blur_sigma
        ).astype(np.float32) / 255.0
        
        if debug:
            print(f"  Threshold={threshold:.2f} - After blur - Min: {heatmap_blurred.min():.4f}, Max: {heatmap_blurred.max():.4f}, Mean: {heatmap_blurred.mean():.4f}")
        
        # Greedy peak detection
        peaks = []
        confidences = []
        heatmap_work = heatmap_blurred.copy()
        
        while True:
            # Find global maximum
            max_idx = np.argmax(heatmap_work)
            y, x = np.unravel_index(max_idx, heatmap_work.shape)
            peak_conf = heatmap_blurred[y, x]
            
            # Stop if below threshold or no valid pixels
            if peak_conf < threshold:
                if debug and len(peaks) == 0:
                    print(f"  No peaks found above threshold {threshold:.2f}")
                break
            
            peaks.append((y, x))
            confidences.append(peak_conf)
            
            # Suppress neighbors
            y_min = max(0, y - min_distance)
            y_max = min(heatmap_work.shape[0], y + min_distance + 1)
            x_min = max(0, x - min_distance)
            x_max = min(heatmap_work.shape[1], x + min_distance + 1)
            heatmap_work[y_min:y_max, x_min:x_max] = 0
            
            # Safety limit
            if len(peaks) > 500:
                if debug:
                    print(f"  WARNING: Stopped at 500 peaks")
                break
        
        if debug:
            print(f"  Peaks detected: {len(peaks)}")
            if len(confidences) > 0:
                print(f"  Confidence stats (above threshold {threshold:.2f}): min={np.min(confidences):.4f}, max={np.max(confidences):.4f}, mean={np.mean(confidences):.4f}")
        
        return peaks, confidences
    
    def detect_peaks_topk(self, heatmap, min_distance=30, blur_sigma=2.0, top_k=None, debug=True):
        """
        Detect mitosis peaks using NMS + confidence filtering (keep top-K peaks).
        
        Parameters
        ----------
        heatmap : np.ndarray
            Model confidence heatmap [H, W] with values [0, 1]
        min_distance : int
            Minimum distance between peaks in pixels (default: 30)
        blur_sigma : float
            Gaussian blur sigma (default: 2.0)
        top_k : int
            Keep only top-K peaks by confidence (default: None, keep all)
        debug : bool
            Print intermediate statistics (default: True)
        
        Returns
        -------
        peaks : list of tuples
            List of (y, x) peak coordinates
        confidences : list of floats
            Confidence value at each peak
        """
        # Normalize heatmap
        heatmap_norm = np.clip(heatmap, 0, 1)
        
        # Apply Gaussian blur
        heatmap_blurred = cv2.GaussianBlur(
            (heatmap_norm * 255).astype(np.uint8),
            ksize=(int(blur_sigma * 2) * 2 + 1, int(blur_sigma * 2) * 2 + 1),
            sigmaX=blur_sigma,
            sigmaY=blur_sigma
        ).astype(np.float32) / 255.0
        
        # Non-maximum suppression
        footprint_size = min_distance * 2 + 1
        heatmap_local_max = maximum_filter(
            heatmap_blurred,
            size=(footprint_size, footprint_size)
        ) == heatmap_blurred
        
        # Get all local maxima
        peaks_y, peaks_x = np.where(heatmap_local_max)
        peaks = [(y, x) for y, x in zip(peaks_y, peaks_x)]
        confidences = [heatmap_blurred[y, x] for y, x in peaks]
        
        if debug:
            print(f"  Local maxima found: {len(peaks)}")
            if len(confidences) > 0:
                print(f"  Confidence stats: min={np.min(confidences):.4f}, max={np.max(confidences):.4f}, mean={np.mean(confidences):.4f}, median={np.median(confidences):.4f}")
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        
        # Keep top-K if specified
        if top_k is not None and len(sorted_indices) > top_k:
            sorted_indices = sorted_indices[:top_k]
            if debug:
                print(f"  Keeping top-{top_k} peaks by confidence (threshold={confidences[sorted_indices[-1]]:.4f})")
        
        peaks_filtered = [peaks[i] for i in sorted_indices]
        confidences_filtered = [confidences[i] for i in sorted_indices]
        
        return peaks_filtered, confidences_filtered

    
    def cluster_peaks(self, peaks, confidences, cluster_distance=30, debug=True):
        """
        Cluster nearby peaks and return cluster centroids weighted by confidence.
        
        Parameters
        ----------
        peaks : list of tuples
            List of (y, x) peak coordinates
        confidences : list of floats
            Confidence values at each peak
        cluster_distance : int
            Maximum distance to merge peaks (default: 30px)
        debug : bool
            Print diagnostics
        
        Returns
        -------
        clustered_peaks : list of tuples
            Cluster centroids (y, x)
        clustered_confidences : list of floats
            Max confidence in each cluster
        """
        if len(peaks) == 0:
            return [], []
        
        peaks_array = np.array(peaks, dtype=np.float32)
        confidences_array = np.array(confidences, dtype=np.float32)
        
        # Use hierarch greedy clustering to merge nearby peaks
        # Compute pairwise distances
        distances = cdist(peaks_array, peaks_array, metric='euclidean')
        
        # Greedy clustering: start with highest confidence, merge all nearby peaks
        sorted_indices = np.argsort(confidences_array)[::-1]  # Descending confidence
        
        clusters = []
        used = set()
        
        for idx in sorted_indices:
            if idx in used:
                continue
            
            # Find all peaks close to this one
            cluster_mask = distances[idx] <= cluster_distance
            cluster_indices = np.where(cluster_mask)[0]
            cluster_indices = [i for i in cluster_indices if i not in used]
            
            if len(cluster_indices) > 0:
                # Use weighted centroid by confidence
                cluster_peaks = peaks_array[cluster_indices]
                cluster_confs = confidences_array[cluster_indices]
                
                # Weighted centroid
                weights = cluster_confs / np.sum(cluster_confs)
                centroid_y = np.sum(cluster_peaks[:, 0] * weights)
                centroid_x = np.sum(cluster_peaks[:, 1] * weights)
                max_conf = np.max(cluster_confs)
                
                clusters.append(((centroid_y, centroid_x), max_conf))
                used.update(cluster_indices)
        
        clustered_peaks = [p for p, c in clusters]
        clustered_confidences = [c for p, c in clusters]
        
        if debug:
            print(f"  Merged {len(peaks)} peaks into {len(clustered_peaks)} clusters (cluster_distance={cluster_distance}px)")
        
        return clustered_peaks, clustered_confidences
    
    def match_predictions_to_gt(self, predicted_peaks, gt_peaks, max_distance=20):
        """
        Match predicted peaks to ground truth using bipartite matching.
        
        Parameters
        ----------
        predicted_peaks : list of tuples
            List of (y, x) predicted peak coordinates
        gt_peaks : list of tuples
            List of (y, x) ground truth peak coordinates
        max_distance : float
            Maximum distance for a match (default: 20 pixels)
        
        Returns
        -------
        matches : list of tuples
            List of (pred_idx, gt_idx) matches
        """
        if len(predicted_peaks) == 0 or len(gt_peaks) == 0:
            return []
        
        # Convert to numpy arrays
        pred_array = np.array(predicted_peaks, dtype=np.float32)
        gt_array = np.array(gt_peaks, dtype=np.float32)
        
        # Compute pairwise distances
        distances = cdist(pred_array, gt_array, metric='euclidean')
        
        # Greedy matching: match closest pairs within max_distance
        matches = []
        matched_pred = set()
        matched_gt = set()
        
        # Sort by distance
        for pred_idx, gt_idx in sorted(
            zip(*np.where(distances <= max_distance)),
            key=lambda x: distances[x]
        ):
            if pred_idx not in matched_pred and gt_idx not in matched_gt:
                matches.append((pred_idx, gt_idx))
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
        
        return matches
    
    def calculate_metrics(self, predicted_peaks, gt_peaks, matches, match_distance=40):
        """
        Calculate detection metrics.
        
        Parameters
        ----------
        predicted_peaks : list
            Predicted peak coordinates
        gt_peaks : list
            Ground truth peak coordinates
        matches : list
            List of (pred_idx, gt_idx) matches
        match_distance : float
            Maximum matching distance used (for diagnostics)
        
        Returns
        -------
        metrics : dict
            Precision, recall, F1, TP, FP, FN
        """
        tp = len(matches)
        fp = len(predicted_peaks) - tp
        fn = len(gt_peaks) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # DIAGNOSTIC: compute closest distance for each prediction and GT to understand mismatch
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_predictions': len(predicted_peaks),
            'total_gt': len(gt_peaks),
            'match_distance': match_distance
        }
        
        # Compute closest distance from each GT to any prediction
        if len(predicted_peaks) > 0 and len(gt_peaks) > 0:
            pred_array = np.array(predicted_peaks, dtype=np.float32)
            gt_array = np.array(gt_peaks, dtype=np.float32)
            distances = cdist(pred_array, gt_array, metric='euclidean')
            
            # For each GT, find closest prediction
            closest_distances_to_pred = np.min(distances, axis=0)
            metrics['gt_closest_pred_distance_mean'] = np.mean(closest_distances_to_pred)
            metrics['gt_closest_pred_distance_max'] = np.max(closest_distances_to_pred)
            metrics['gt_closest_pred_distance_median'] = np.median(closest_distances_to_pred)
            
            # For each prediction, find closest GT
            closest_distances_to_gt = np.min(distances, axis=1)
            metrics['pred_closest_gt_distance_mean'] = np.mean(closest_distances_to_gt)
            metrics['pred_closest_gt_distance_max'] = np.max(closest_distances_to_gt)
            metrics['pred_closest_gt_distance_median'] = np.median(closest_distances_to_gt)
        
        return metrics
    
    def detect_and_evaluate(self, image_path, threshold=0.5, min_distance=30, max_match_distance=20):
        """
        Full pipeline: classify image, detect peaks, match with GT, calculate metrics.
        
        Parameters
        ----------
        image_path : str or Path
            Path to WSI image
        threshold : float
            Peak detection threshold
        min_distance : int
            Minimum distance between peaks
        max_match_distance : float
            Maximum distance for peak matching
        
        Returns
        -------
        results : dict
            Heatmap, peaks, GT peaks, matches, metrics
        """
        # Classify image to get heatmap
        print("\nClassifying image...")
        heatmap, predictions, confidences, positions, image, mask = self.classifier.classify_wsi(
            image_path, threshold=0.5  # Use 0.5 for confidence, not for peak detection
        )
        
        # DEBUG: Print heatmap statistics
        print(f"\n--- HEATMAP STATISTICS ---")
        print(f"Shape: {heatmap.shape}")
        print(f"Min: {heatmap.min():.4f}, Max: {heatmap.max():.4f}")
        print(f"Mean: {heatmap.mean():.4f}, Median: {np.median(heatmap):.4f}")
        print(f"Std: {heatmap.std():.4f}")
        print(f"Percentiles: 25%={np.percentile(heatmap, 25):.4f}, 75%={np.percentile(heatmap, 75):.4f}, 95%={np.percentile(heatmap, 95):.4f}")
        
        # Detect peaks from heatmap
        print(f"\nDetecting mitosis peaks (threshold={threshold}, min_distance={min_distance})...")
        predicted_peaks, pred_confidences = self.detect_peaks(
            heatmap, 
            threshold=threshold,
            min_distance=min_distance
        )
        
        # Load ground truth if available
        gt_peaks = []
        if mask is not None:
            # Convert mask to binary and find GT peak locations
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # Dilate mask slightly to connect nearby components
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
            
            # Find contours and get centroids
            contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cy = int(M['m01'] / M['m00'])
                    cx = int(M['m10'] / M['m00'])
                    gt_peaks.append((cy, cx))
        
        # Match predictions to GT
        print(f"Matching predictions to ground truth (max_distance={max_match_distance})...")
        matches = self.match_predictions_to_gt(
            predicted_peaks, 
            gt_peaks, 
            max_distance=max_match_distance
        )
        
        # Calculate metrics
        metrics = self.calculate_metrics(predicted_peaks, gt_peaks, matches)
        
        return {
            'image': image,
            'heatmap': heatmap,
            'mask': mask,
            'predicted_peaks': predicted_peaks,
            'pred_confidences': pred_confidences,
            'gt_peaks': gt_peaks,
            'matches': matches,
            'metrics': metrics,
            'threshold': threshold,
            'min_distance': min_distance
        }
    
    def visualize_detections(self, results, output_path):
        """
        Visualize mitosis detection results.
        
        Parameters
        ----------
        results : dict
            Results from detect_and_evaluate()
        output_path : str or Path
            Path to save visualizations
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image = results['image']
        heatmap = results['heatmap']
        mask = results['mask']
        predicted_peaks = results['predicted_peaks']
        gt_peaks = results['gt_peaks']
        matches = results['matches']
        metrics = results['metrics']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        # Panel 1: Original image with predictions
        ax = axes[0, 0]
        ax.imshow(image)
        ax.plot([x for y, x in predicted_peaks], [y for y, x in predicted_peaks], 'r+', markersize=15, markeredgewidth=2)
        ax.plot([x for y, x in gt_peaks], [y for y, x in gt_peaks], 'go', markersize=8, markerfacecolor='none', markeredgewidth=2)
        ax.set_title(f"Predicted (Red +) vs Ground Truth (Green O)\nPrecision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1']:.2f}", 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Panel 2: Heatmap with peaks
        ax = axes[0, 1]
        im = ax.imshow(heatmap, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.plot([x for y, x in predicted_peaks], [y for y, x in predicted_peaks], 'w+', markersize=12, markeredgewidth=2)
        ax.set_title(f"Heatmap + Detected Peaks\nThreshold: {results['threshold']:.2f}, Min Dist: {results['min_distance']}", 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Confidence')
        
        # Panel 3: Ground truth mask
        if mask is not None:
            ax = axes[1, 0]
            ax.imshow(mask, cmap='gray')
            ax.plot([x for y, x in gt_peaks], [y for y, x in gt_peaks], 'r+', markersize=15, markeredgewidth=2)
            ax.set_title(f"Ground Truth Mask + Peak Centers\nTotal GT: {len(gt_peaks)}", 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'No ground truth mask available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # Panel 4: Metrics summary
        ax = axes[1, 1]
        ax.axis('off')
        
        metrics_text = f"""
DETECTION METRICS
═════════════════════════

Precision:  {metrics['precision']:.4f}
Recall:     {metrics['recall']:.4f}
F1 Score:   {metrics['f1']:.4f}

─────────────────────────

True Positives (TP):    {metrics['tp']}
False Positives (FP):   {metrics['fp']}
False Negatives (FN):   {metrics['fn']}

─────────────────────────

Total Predictions:  {metrics['total_predictions']}
Total Ground Truth:  {metrics['total_gt']}

═════════════════════════

Threshold:      {results['threshold']:.2f}
Min Distance:   {results['min_distance']}
Max Match Dist: 60 px
        """
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save figure
        output_file = output_path / "mitosis_detection_detailed.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Visualization saved: {output_file}")


def optimize_threshold(detector, image_path, masks_dir, thresholds=None, min_distance=30):
    """
    Find optimal threshold by testing on training/validation set.
    
    Parameters
    ----------
    detector : MitosisDetector
        Detector instance
    image_path : str or Path
        Path to image
    masks_dir : str or Path
        Masks directory
    thresholds : list
        List of thresholds to test (default: 0.3-0.8)
    min_distance : int
        Minimum distance between peaks (default: 30)
    
    Returns
    -------
    best_threshold : float
        Threshold with highest F1 score
    results_by_threshold : dict
        Metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.8, 0.05)
    
    results_by_threshold = {}
    
    print(f"\nOptimizing threshold on {Path(image_path).name}...")
    print("=" * 80)
    
    for threshold in thresholds:
        results = detector.detect_and_evaluate(
            image_path, 
            threshold=threshold,
            min_distance=min_distance
        )
        metrics = results['metrics']
        results_by_threshold[threshold] = metrics
        
        print(f"Threshold {threshold:.2f}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f} "
              f"(TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']})")
    
    # Find best threshold
    best_threshold = max(results_by_threshold.keys(), 
                        key=lambda t: results_by_threshold[t]['f1'])
    
    print("=" * 80)
    print(f"\n✓ Best threshold: {best_threshold:.2f} (F1={results_by_threshold[best_threshold]['f1']:.4f})")
    
    return best_threshold, results_by_threshold


def main():
    """Interactive mitosis detection workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect mitosis locations in WSI')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Model directory (models, models_256, etc.). Default: models')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MITOSIS DETECTION & LOCALIZATION")
    print("="*80)
    
    # Determine model directory
    solution_dir = Path(__file__).parent
    if args.model_dir is None:
        model_dir = solution_dir / "models"
    else:
        model_dir = Path(args.model_dir)
    
    print(f"Using model directory: {model_dir}")
    
    # Hardcoded directories
    image_dir = Path(r"C:\DIPLOMSKI\MIDOG_Challenge\images")
    masks_dir = Path(solution_dir / "data" / "masks")  # Use 128px masks by default (both work since masks don't change with patch size)
    
    # Pick image
    print(f"\nImage directory: {image_dir}")
    image_path = pick_image_interactive(image_dir)
    if image_path is None:
        return
    
    print(f"✓ Selected: {image_path.name}")
    
    # Pick model from the specified directory
    model_path = pick_model_interactive(model_dir=str(model_dir))
    if model_path is None:
        return
    
    print(f"✓ Selected: {Path(model_path).stem}")
    
    # Auto-detect patch size from model directory
    patch_size = 128  # default
    stride = 128       # default
    
    # Check if model_dir contains 'models_XXX' pattern (e.g., models_256, models_384)
    for part in Path(model_dir).parts:
        if part.startswith('models_'):
            try:
                patch_size = int(part.replace('models_', ''))
                # Stride should be roughly half the patch size for good coverage
                stride = max(64, patch_size // 2)
                break
            except ValueError:
                pass
    
    print(f"Detected patch size: {patch_size}×{patch_size}, stride: {stride}px")
    
    # Initialize classifier and detector
    print("\nInitializing classifier...")
    classifier = WSIClassifier(model_path, device='cuda', patch_size=patch_size, stride=stride, masks_dir=masks_dir)
    detector = MitosisDetector(classifier)
    
    # Classify image to get heatmap
    print("\nClassifying image to get heatmap...")
    heatmap, predictions, confidences, positions, image, mask = classifier.classify_wsi(
        image_path, threshold=0.5
    )
    
    # Optimize threshold by testing multiple values
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION")
    print("="*80)
    best_threshold, results_by_threshold = optimize_threshold(
        detector, 
        image_path, 
        masks_dir,
        thresholds=np.arange(0.3, 0.85, 0.05),
        min_distance=80
    )
    
    # Run final detection with best threshold
    print("\n" + "="*80)
    print("FINAL DETECTION")
    print("="*80)
    
    print(f"\nRunning detection with optimized threshold: {best_threshold:.2f}")
    predicted_peaks, pred_confidences = detector.detect_peaks_greedy(
        heatmap,
        threshold=best_threshold,
        min_distance=80,
        blur_sigma=2.0,
        debug=True
    )
    
    # Cluster nearby peaks
    print(f"\nClustering nearby peaks...")
    predicted_peaks, pred_confidences = detector.cluster_peaks(
        predicted_peaks,
        pred_confidences,
        cluster_distance=170,
        debug=True
    )
    
    # Load ground truth and calculate metrics
    gt_peaks = []
    if mask is not None:
        mask_binary = (mask > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
        contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cy = int(M['m01'] / M['m00'])
                cx = int(M['m10'] / M['m00'])
                gt_peaks.append((cy, cx))
    
    # Match and calculate metrics
    matches = detector.match_predictions_to_gt(predicted_peaks, gt_peaks, max_distance=60)
    metrics = detector.calculate_metrics(predicted_peaks, gt_peaks, matches, match_distance=60)
    
    # Create results dict for visualization
    results = {
        'image': image,
        'heatmap': heatmap,
        'mask': mask,
        'predicted_peaks': predicted_peaks,
        'pred_confidences': pred_confidences,
        'gt_peaks': gt_peaks,
        'matches': matches,
        'metrics': metrics,
        'threshold': best_threshold,
        'min_distance': 80,
        'cluster_distance': 170
    }
    
    # Print results
    print("\n" + "="*80)
    print("DETECTION RESULTS")
    print("="*80)
    print(f"\nPrecision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1 Score:   {metrics['f1']:.4f}")
    print(f"\nTrue Positives:   {metrics['tp']}")
    print(f"False Positives:  {metrics['fp']}")
    print(f"False Negatives:  {metrics['fn']}")
    print(f"\nTotal Predictions: {metrics['total_predictions']}")
    print(f"Total Ground Truth: {metrics['total_gt']}")
    
    # Visualize
    output_dir = Path(__file__).parent / "output" / "mitosis_detection"
    print(f"\nVisualizing results...")
    detector.visualize_detections(results, output_dir)
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
