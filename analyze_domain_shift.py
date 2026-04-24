"""
Domain shift analysis: Test all models on all scanner test sets
=====================================================================
Evaluates all 12 model combinations (3 scanners × 4 augmentations)
against 3 scanner test sets to measure domain shift degradation
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, str(Path(__file__).parent))

from patch_classifier_model import PatchClassifier
from patch_classifier_dataset_augmented import PatchClassifierDatasetAugmented


def evaluate_model_on_scanner(model, test_loader, device):
    """Evaluate model on test loader"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate TP, FP, TN, FN
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'n_samples': len(all_labels),
        'n_positive': int(np.sum(all_labels)),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }
    
    return metrics


def get_scanner_selection():
    """
    Interactive menu to select which scanner models to evaluate
    Returns list of scanners selected
    """
    scanners = ["Hamamatsu_XR", "Hamamatsu_S360", "Aperio_CS"]
    
    print(f"\n{'='*100}")
    print("SELECT TRAINING SCANNER MODELS TO ANALYZE")
    print(f"{'='*100}")
    print("1. Hamamatsu_XR")
    print("2. Hamamatsu_S360")
    print("3. Aperio_CS")
    print("4. All Scanners")
    print(f"{'='*100}\n")
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            return [scanners[0]]
        elif choice == "2":
            return [scanners[1]]
        elif choice == "3":
            return [scanners[2]]
        elif choice == "4":
            return scanners
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def get_test_scanner_selection():
    """
    Interactive menu to select which scanner test sets to analyze
    Returns list of test scanners selected
    """
    scanners = ["Hamamatsu_XR", "Hamamatsu_S360", "Aperio_CS"]
    
    print(f"\n{'='*100}")
    print("SELECT TEST SCANNER SETS TO ANALYZE")
    print(f"{'='*100}")
    print("1. Hamamatsu_XR")
    print("2. Hamamatsu_S360")
    print("3. Aperio_CS")
    print("4. All Test Scanners")
    print(f"{'='*100}\n")
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            return [scanners[0]]
        elif choice == "2":
            return [scanners[1]]
        elif choice == "3":
            return [scanners[2]]
        elif choice == "4":
            return scanners
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def get_augmentation_selection():
    """
    Interactive menu to select which augmentation strategies to analyze
    Returns list of augmentations selected
    """
    augmentations = ["standard", "medium", "strong", "histology"]
    
    print(f"\n{'='*100}")
    print("SELECT AUGMENTATION STRATEGIES TO ANALYZE")
    print(f"{'='*100}")
    print("1. Standard")
    print("2. Medium")
    print("3. Strong")
    print("4. Histology")
    print("5. All Augmentations")
    print(f"{'='*100}\n")
    
    while True:
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == "1":
            return [augmentations[0]]
        elif choice == "2":
            return [augmentations[1]]
        elif choice == "3":
            return [augmentations[2]]
        elif choice == "4":
            return [augmentations[3]]
        elif choice == "5":
            return augmentations
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


def analyze_domain_shift(data_root, model_dir, output_dir, selected_train_scanners=None, selected_test_scanners=None, selected_augmentations=None):
    """
    Test selected models on selected scanner test sets
    """
    
    data_root = Path(data_root)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scanners = ["Hamamatsu_XR", "Hamamatsu_S360", "Aperio_CS"]
    augmentations = ["standard", "medium", "strong", "histology"]
    
    # Use selected scanners/augmentations or default to all
    if selected_train_scanners is None:
        selected_train_scanners = scanners
    if selected_test_scanners is None:
        selected_test_scanners = scanners
    if selected_augmentations is None:
        selected_augmentations = augmentations
    
    # Collect all results
    results = []
    models_found = 0
    models_missing = 0
    
    print(f"\n{'='*100}")
    print("DOMAIN SHIFT ANALYSIS - Testing Selected Models on Selected Scanner Test Sets")
    print(f"{'='*100}\n")
    print(f"Training Scanners: {', '.join(selected_train_scanners)}")
    print(f"Test Scanners: {', '.join(selected_test_scanners)}")
    print(f"Augmentations: {', '.join(selected_augmentations)}")
    print(f"{'='*100}\n")
    
    # Load each model and test on selected scanners
    for train_scanner in selected_train_scanners:
        for aug in selected_augmentations:
            model_name = f"patch_classifier_{train_scanner.replace(' ', '_')}_{aug}.pth"
            model_path = model_dir / model_name
            
            if not model_path.exists():
                print(f"Model not found: {model_path}")
                models_missing += 1
                continue
            
            models_found += 1
            
            # Load model
            model = PatchClassifier().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            print(f"Testing model: {train_scanner} + {aug}")
            print("-" * 100)
            
            # Test on selected scanners
            for test_scanner in selected_test_scanners:
                test_dataset = PatchClassifierDatasetAugmented(
                    root=data_root,
                    scanner=test_scanner,
                    augmentation=None,
                    split='test'
                )
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                metrics = evaluate_model_on_scanner(model, test_loader, device)
                
                same_domain = "SAME" if train_scanner == test_scanner else "DIFF"
                
                print(f"  Test on {test_scanner:20s} {same_domain:10s} | "
                      f"Acc: {metrics['accuracy']:.4f} | "
                      f"F1: {metrics['f1']:.4f} | "
                      f"AUC: {metrics['auc']:.4f} | "
                      f"TP: {metrics['tp']} | FP: {metrics['fp']} | "
                      f"TN: {metrics['tn']} | FN: {metrics['fn']}")
                
                results.append({
                    'train_scanner': train_scanner,
                    'augmentation': aug,
                    'test_scanner': test_scanner,
                    'same_domain': train_scanner == test_scanner,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'auc': metrics['auc'],
                    'n_samples': metrics['n_samples'],
                    'n_positive': metrics['n_positive'],
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'tn': metrics['tn'],
                    'fn': metrics['fn']
                })
            
            print()
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = output_dir / "domain_shift_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}\n")
    
    # Create visualizations
    print("Generating visualizations...")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Heatmap: F1 scores for each model-scanner combination
    # Create separate heatmap for each augmentation
    
    aug_translations = {
        'standard': 'Standardna augmentacija',
        'medium': 'Srednja augmentacija',
        'strong': 'Jaka augmentacija',
        'histology': 'Histološka augmentacija'
    }
    
    for aug in selected_augmentations:
        aug_data = results_df[results_df['augmentation'] == aug]
        
        if aug_data.empty:
            continue
        
        # Create pivot table
        pivot = aug_data.pivot_table(
            values='f1',
            index='train_scanner',
            columns='test_scanner',
            aggfunc='mean'
        )
        
        # Reorder to match selected scanners
        pivot = pivot.reindex(selected_train_scanners, axis=0).reindex(selected_test_scanners, axis=1)
        
        # Create individual figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.0, vmax=0.95,
                   ax=ax, cbar_kws={'label': 'F1 mjera'})
        
        ax.set_title(f'{aug_translations[aug]}', fontweight='bold', fontsize=14)
        ax.set_xlabel('Skener na kojem je testiran', fontweight='bold')
        ax.set_ylabel('Skener na kojem je treniran', fontweight='bold')
        
        plt.tight_layout()
        
        # Save individual heatmap
        heatmap_path = images_dir / f"f1_heatmap_{aug}.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved: {heatmap_path}")
        plt.close()
    
    # 2. Bar plot: Same-domain vs cross-domain performance
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter for selected augmentations only
    selected_results = results_df[results_df['augmentation'].isin(selected_augmentations)]
    same_domain = selected_results[selected_results['same_domain'] == True].groupby('augmentation')['f1'].mean()
    diff_domain = selected_results[selected_results['same_domain'] == False].groupby('augmentation')['f1'].mean()
    
    # Ensure order matches selected_augmentations
    same_domain = same_domain.reindex(selected_augmentations)
    diff_domain = diff_domain.reindex(selected_augmentations)
    
    x = np.arange(len(selected_augmentations))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, same_domain.values, width, label='Ista domena (Unutar domene)', 
                   color='green', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, diff_domain.values, width, label='Različita domena (Van domene)',
                   color='red', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Vrsta augmentacije', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prosječna F1 mjera', fontsize=12, fontweight='bold')
    ax.set_title('Utjecaj domenskog pomaka: Performanse modela unutar i izvan domene', 
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    
    # Croatian translations for augmentations
    aug_labels = {
        'standard': 'Standardna',
        'medium': 'Srednja',
        'strong': 'Jaka',
        'histology': 'Histološka'
    }
    ax.set_xticklabels([aug_labels.get(a, a.capitalize()) for a in selected_augmentations])
    
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.set_ylim([0.0, 0.95])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height < 0.05:
                label_y = 0.02
                va_align = 'bottom'
                fontsize_label = 9
                fontweight_label = 'bold'
                color_label = 'red' if bar.get_facecolor() == (1.0, 0.0, 0.0, 0.7) else 'black'
            else:
                label_y = height
                va_align = 'bottom'
                fontsize_label = 10
                fontweight_label = 'bold'
                color_label = 'black'
            
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{height:.3f}',
                   ha='center', va=va_align, fontsize=fontsize_label, fontweight=fontweight_label, color=color_label)
    
    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.08, right=0.95)
    domain_path = images_dir / "domain_shift_comparison.png"
    plt.savefig(domain_path, dpi=150, bbox_inches='tight')
    print(f"Domain comparison saved: {domain_path}")
    plt.close()
    
    # 3. Summary statistics
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}\n")
    
    for aug in selected_augmentations:
        aug_data = results_df[results_df['augmentation'] == aug]
        
        if aug_data.empty:
            print(f"{aug.upper()} Augmentation: NO DATA (models may not exist)")
            print()
            continue
        
        same = aug_data[aug_data['same_domain'] == True]
        diff = aug_data[aug_data['same_domain'] == False]
        
        print(f"{aug.upper()} Augmentation:")
        print(f"  In-Domain (Same Scanner):")
        if not same.empty:
            f1_std = same['f1'].std() if len(same) > 1 else 0.0
            auc_std = same['auc'].std() if len(same) > 1 else 0.0
            print(f"    F1:  {same['f1'].mean():.4f} ± {f1_std:.4f}")
            print(f"    AUC: {same['auc'].mean():.4f} ± {auc_std:.4f}")
        else:
            print(f"    F1:  N/A ± N/A")
            print(f"    AUC: N/A ± N/A")
        
        print(f"  Cross-Domain (Different Scanner):")
        if not diff.empty:
            f1_std = diff['f1'].std() if len(diff) > 1 else 0.0
            auc_std = diff['auc'].std() if len(diff) > 1 else 0.0
            print(f"    F1:  {diff['f1'].mean():.4f} ± {f1_std:.4f}")
            print(f"    AUC: {diff['auc'].mean():.4f} ± {auc_std:.4f}")
            print(f"  Domain Gap (In - Cross): {(same['f1'].mean() - diff['f1'].mean()):.4f}")
        else:
            print(f"    F1:  N/A ± N/A")
            print(f"    AUC: N/A ± N/A")
            print(f"  Domain Gap (In - Cross): N/A")
        print()
    
    print(f"{'='*100}")
    print(f"MODELS SUMMARY: {models_found} found, {models_missing} missing")
    print(f"{'='*100}\n")
    
    if models_missing > 0:
        print(f"Note: {models_missing} model(s) were not found.")
        print(f"   Train them using: python train_domain_shift.py")
        print(f"   (Missing models result in N/A statistics for cross-domain comparisons)\n")
    
    if results_df.empty:
        print("No results to analyze. Please check that models exist in ./models/\n")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze domain shift across scanners')
    parser.add_argument('--data-root', type=str, default=None,
                       help='Path to patch data (auto-detects 128px vs 256px). '
                            'Default: data/patches/multi_scanner (128px) or data_256/patches/multi_scanner (256px)')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Path to models directory (auto-detected from data-root if not provided)')
    args = parser.parse_args()
    
    solution_dir = Path(__file__).parent
    
    # Auto-detect or use provided data-root
    if args.data_root is None:
        data_root = solution_dir / "data" / "patches" / "multi_scanner"
    else:
        data_root = Path(args.data_root)
    
    # Auto-detect model directory based on data root
    if args.model_dir is None:
        # Check if data_root contains 'data_XXX' pattern
        if 'data_' in str(data_root):
            for part in data_root.parts:
                if part.startswith('data_'):
                    patch_size = part.replace('data_', '')
                    args.model_dir = str(solution_dir / f"models_{patch_size}")
                    break
        
        # Default to models/ if no patch size detected
        if args.model_dir is None:
            args.model_dir = str(solution_dir / "models")
    
    model_dir = Path(args.model_dir)
    output_dir = solution_dir / "domain_shift_results"
    
    print(f"\nUsing:")
    print(f"  Data root: {data_root}")
    print(f"  Model dir: {model_dir}")
    print(f"  Output dir: {output_dir}\n")
    
    if data_root.exists():
        # Get user selections
        selected_train_scanners = get_scanner_selection()
        selected_test_scanners = get_test_scanner_selection()
        selected_augmentations = get_augmentation_selection()
        
        results_df = analyze_domain_shift(
            data_root=data_root,
            model_dir=model_dir,
            output_dir=output_dir,
            selected_train_scanners=selected_train_scanners,
            selected_test_scanners=selected_test_scanners,
            selected_augmentations=selected_augmentations
        )
        print("Analysis complete!")
    else:
        print(f"Error: Data root not found: {data_root}")
        print("Please ensure patches exist at the data root path")
