"""
Out-of-Domain (OOD) Metrics and ROC Curves Analysis
====================================================
Generates table with precision/recall metrics for out-of-domain detection
and visualizes ROC curves for all models evaluated on different scanners.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, str(Path(__file__).parent))

from patch_classifier_model import PatchClassifier
from patch_classifier_dataset_augmented import PatchClassifierDatasetAugmented


def evaluate_model_on_scanner_detailed(model, test_loader, device):
    """
    Evaluate model on test loader and return all necessary metrics
    Returns predictions, labels, probabilities for ROC/PR curves
    """
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
    
    # Calculate precision and recall for both classes
    # Positive class (mitosis): class 1
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Negative class (background): class 0
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'precision_pos': precision_pos,
        'recall_pos': recall_pos,
        'precision_neg': precision_neg,
        'recall_neg': recall_neg,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }
    
    return metrics


def generate_ood_analysis(data_root, model_dir, output_dir):
    """
    Analyze out-of-domain performance:
    - Generate metrics table (precision/recall for both classes)
    - Plot ROC curves for all OOD combinations
    """
    
    data_root = Path(data_root)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    scanners = ["Hamamatsu_XR", "Hamamatsu_S360", "Aperio_CS"]
    augmentations = ["standard", "medium", "strong", "histology"]
    
    # Collect results for table and ROC curves
    results_list = []
    roc_data = {}  # Store ROC data for plotting
    
    print(f"{'='*120}")
    print("OUT-OF-DOMAIN EVALUATION - All Models on All Test Scanners")
    print(f"{'='*120}\n")
    
    # Evaluate each model on all test scanners
    for train_scanner in scanners:
        for aug in augmentations:
            model_name = f"patch_classifier_{train_scanner.replace(' ', '_')}_{aug}.pth"
            model_path = model_dir / model_name
            
            if not model_path.exists():
                print(f"⚠️  Model not found: {model_path}")
                continue
            
            # Load model
            model = PatchClassifier().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            print(f"Evaluating: {train_scanner} + {aug}")
            print("-" * 120)
            
            # Test on all scanners
            for test_scanner in scanners:
                test_dataset = PatchClassifierDatasetAugmented(
                    root=data_root,
                    scanner=test_scanner,
                    augmentation=None,
                    split='test'
                )
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                metrics = evaluate_model_on_scanner_detailed(model, test_loader, device)
                
                is_same_domain = train_scanner == test_scanner
                domain_type = "IN-DOMAIN" if is_same_domain else "OUT-OF-DOMAIN"
                
                print(f"  Test on {test_scanner:20s} [{domain_type}] | "
                      f"Prec(+): {metrics['precision_pos']:.4f} | "
                      f"Rec(+): {metrics['recall_pos']:.4f} | "
                      f"Prec(-): {metrics['precision_neg']:.4f} | "
                      f"Rec(-): {metrics['recall_neg']:.4f}")
                
                # Store results for table
                model_key = f"{train_scanner}_{aug}"
                results_list.append({
                    'model_key': model_key,
                    'train_scanner': train_scanner,
                    'augmentation': aug,
                    'test_scanner': test_scanner,
                    'domain_type': domain_type,
                    'precision_positive': metrics['precision_pos'],
                    'recall_positive': metrics['recall_pos'],
                    'precision_negative': metrics['precision_neg'],
                    'recall_negative': metrics['recall_neg'],
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'tn': metrics['tn'],
                    'fn': metrics['fn']
                })
                
                # Store ROC data for out-of-domain samples only
                if not is_same_domain:
                    model_label = f"{train_scanner}_{aug}"
                    if model_label not in roc_data:
                        roc_data[model_label] = []
                    
                    roc_data[model_label].append({
                        'test_scanner': test_scanner,
                        'labels': metrics['labels'],
                        'probs': metrics['probs']
                    })
            
            print()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Save full results
    csv_path = output_dir / "ood_metrics_full.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Full results saved to: {csv_path}\n")
    print(f"{'='*150}")
    print("OUT-OF-DOMAIN METRICS TABLE (Precision and Recall for Both Classes)")
    print(f"{'='*150}\n")
    
    ood_results = results_df[results_df['domain_type'] == 'OUT-OF-DOMAIN'].copy()
    
    # Create display table with model info and metrics
    display_table = ood_results[[
        'model_key',
        'test_scanner',
        'precision_positive',
        'recall_positive',
        'precision_negative',
        'recall_negative'
    ]].copy()
    
    display_table.columns = [
        'Model',
        'Test Scanner',
        'Precision (+)',
        'Recall (+)',
        'Precision (-)',
        'Recall (-)'
    ]
    
    # Round values to 4 decimals for display
    for col in ['Precision (+)', 'Recall (+)', 'Precision (-)', 'Recall (-)']:
        display_table[col] = display_table[col].apply(lambda x: f'{x:.4f}')
    
    print(display_table.to_string(index=False))
    print()
    
    # Save summary table with numeric values
    summary_path = output_dir / "ood_metrics_summary.csv"
    display_table_numeric = ood_results[[
        'model_key',
        'test_scanner',
        'precision_positive',
        'recall_positive',
        'precision_negative',
        'recall_negative'
    ]].copy()
    display_table_numeric.columns = [
        'Model',
        'Test_Scanner',
        'Precision_Positive_Class',
        'Recall_Positive_Class',
        'Precision_Negative_Class',
        'Recall_Negative_Class'
    ]
    display_table_numeric.to_csv(summary_path, index=False)
    print(f"✅ Summary table saved to: {summary_path}\n")
    
    # Create aggregated summary table (average metrics per model across all OOD test scanners)
    print(f"\n{'='*120}")
    print("EVALUACIJA VAN DOMENE - METRIKE")
    print("Prosječna preciznosti i odziv (na svim testnim skenerima)")
    print(f"{'='*120}\n")
    
    # Calculate aggregated metrics
    agg_numeric = ood_results.groupby('model_key').apply(
        lambda x: pd.Series({
            'Precision_Positive': x['precision_positive'].mean(),
            'Precision_Negative': x['precision_negative'].mean(),
            'Recall': (x['recall_positive'].mean() + x['recall_negative'].mean()) / 2  # Macro recall
        })
    ).reset_index()
    
    agg_numeric.columns = [
        'Model',
        'Precision_Positive',
        'Precision_Negative',
        'Recall'
    ]
    
    # Display table
    agg_display = agg_numeric.copy()
    for col in ['Precision_Positive', 'Precision_Negative', 'Recall']:
        agg_display[col] = agg_display[col].apply(lambda x: f'{x:.4f}')
    
    print(agg_display.to_string(index=False))
    print()
    
    # Save aggregated table for LaTeX
    agg_path = output_dir / "ood_metrics_latex.csv"
    agg_numeric.to_csv(agg_path, index=False)
    print(f"✅ Agregirana tablica (za LaTeX) sprema u: {agg_path}\n")
    
    # Create visual table as image for detailed metrics
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for visual table
    table_data = []
    table_data.append(['Model', 'Test Scanner', 'Prec (+)', 'Rec (+)', 'Prec (-)', 'Rec (-)'])
    
    for _, row in ood_results.iterrows():
        table_data.append([
            row['model_key'][:25],
            row['test_scanner'],
            f"{row['precision_positive']:.4f}",
            f"{row['recall_positive']:.4f}",
            f"{row['precision_negative']:.4f}",
            f"{row['recall_negative']:.4f}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ECF0F1')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title('OUT-OF-DOMAIN EVALUATION - Detailed Metrics\nPrecision and Recall for Positive (+) and Negative (-) Classes', 
              fontsize=14, fontweight='bold', pad=20)
    
    table_image_path = output_dir / "ood_metrics_detailed_table.png"
    plt.savefig(table_image_path, dpi=150, bbox_inches='tight')
    print(f"✅ Detailed metrics table image saved to: {table_image_path}\n")
    plt.close()
    
    # Create visual aggregated table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare aggregated data for visualization
    agg_visual = []
    agg_visual.append(['Model', 'Precision (+)', 'Precision (-)', 'Recall'])
    
    for _, row in agg_numeric.iterrows():
        agg_visual.append([
            row['Model'][:30],
            f"{row['Precision_Positive']:.4f}",
            f"{row['Precision_Negative']:.4f}",
            f"{row['Recall']:.4f}"
        ])
    
    # Create aggregated table
    agg_table = ax.table(cellText=agg_visual, cellLoc='center', loc='center',
                         colWidths=[0.35, 0.20, 0.20, 0.20])
    
    agg_table.auto_set_font_size(False)
    agg_table.set_fontsize(11)
    agg_table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        agg_table[(0, i)].set_facecolor('#E74C3C')
        agg_table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Alternate row colors
    for i in range(1, len(agg_visual)):
        for j in range(4):
            if i % 2 == 0:
                agg_table[(i, j)].set_facecolor('#F8F9FA')
            else:
                agg_table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title('Evaluacija van domene - metrike\nProsječna preciznosti i odziv (na svim testnim skenerima)', 
              fontsize=13, fontweight='bold', pad=20)
    
    agg_table_image_path = output_dir / "ood_metrics_aggregated_table.png"
    plt.savefig(agg_table_image_path, dpi=150, bbox_inches='tight')
    print(f"✅ Vizuelna tablica sprema u: {agg_table_image_path}\n")
    plt.close()
    
    # Generate ROC curves visualization
    print(f"{'='*120}")
    print("GENERATING ROC CURVES - OUT-OF-DOMAIN EVALUATION")
    print(f"{'='*120}\n")
    
    # Combine all OOD data for each model
    combined_roc_data = {}
    for model_label, data_list in roc_data.items():
        all_labels = []
        all_probs = []
        for data in data_list:
            all_labels.extend(data['labels'])
            all_probs.extend(data['probs'])
        
        combined_roc_data[model_label] = {
            'labels': np.array(all_labels),
            'probs': np.array(all_probs)
        }
    
    # Create ROC curves plot
    if combined_roc_data:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors for scanners and linestyles for augmentations
        scanner_colors = {
            'Hamamatsu_XR': '#E74C3C',      # Crvena
            'Hamamatsu_S360': '#3498DB',    # Plava
            'Aperio_CS': '#2ECC71'          # Zelena
        }
        
        aug_linestyles = {
            'standard': '-',
            'medium': '--',
            'strong': '-.',
            'histology': ':'
        }
        
        aug_translations = {
            'standard': 'standardna',
            'medium': 'srednja',
            'strong': 'jaka',
            'histology': 'histološka'
        }
        
        # Plot ROC curves for each model
        for model_label in sorted(combined_roc_data.keys()):
            train_scanner = model_label.split('_')[0]
            if len(model_label.split('_')) > 2:
                train_scanner = '_'.join(model_label.split('_')[:-1])
                aug = model_label.split('_')[-1]
            else:
                parts = model_label.rsplit('_', 1)
                train_scanner = parts[0]
                aug = parts[1]
            
            data = combined_roc_data[model_label]
            labels = data['labels']
            probs = data['probs']
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            
            # Get color and linestyle
            color = scanner_colors.get(train_scanner, '#999999')
            linestyle = aug_linestyles.get(aug, '-')
            aug_display = aug_translations.get(aug, aug)
            
            # Plot
            label = f"{train_scanner} ({aug_display}) AUC={roc_auc:.3f}"
            ax.plot(fpr, tpr, linewidth=2.5, linestyle=linestyle, color=color, label=label)
        
        # Formatting
        ax.set_xlabel('Stopa lažnih pozitiva', fontsize=12, fontweight='bold')
        ax.set_ylabel('Stopa istinitih pozitiva', fontsize=12, fontweight='bold')
        ax.set_title('ROC krivulje - evaluacija van domene', 
                    fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        # Save figure
        roc_path = output_dir / "roc_curves_ood.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved to: {roc_path}")
        plt.close()
        
        # Create separate figure with legend outside for better readability
        fig, ax = plt.subplots(figsize=(13, 9))
        
        for model_label in sorted(combined_roc_data.keys()):
            train_scanner = model_label.split('_')[0]
            if len(model_label.split('_')) > 2:
                train_scanner = '_'.join(model_label.split('_')[:-1])
                aug = model_label.split('_')[-1]
            else:
                parts = model_label.rsplit('_', 1)
                train_scanner = parts[0]
                aug = parts[1]
            
            data = combined_roc_data[model_label]
            labels = data['labels']
            probs = data['probs']
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            
            # Get color and linestyle
            color = scanner_colors.get(train_scanner, '#999999')
            linestyle = aug_linestyles.get(aug, '-')
            aug_display = aug_translations.get(aug, aug)
            
            # Plot
            label = f"{train_scanner} ({aug_display}) - AUC={roc_auc:.3f}"
            ax.plot(fpr, tpr, linewidth=2.5, linestyle=linestyle, color=color, label=label)
        
        # Formatting
        ax.set_xlabel('Stopa lažnih pozitiva', fontsize=12, fontweight='bold')
        ax.set_ylabel('Stopa istinitih pozitiva', fontsize=12, fontweight='bold')
        ax.set_title('ROC krivulje - evaluacija van domene', 
                    fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Legend outside
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, framealpha=0.95)
        
        plt.tight_layout()
        
        # Save figure
        roc_path_legend = output_dir / "roc_curves_ood_legend_outside.png"
        plt.savefig(roc_path_legend, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves with legend outside saved to: {roc_path_legend}")
        plt.close()
    
    print(f"\n{'='*120}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    # Configuration
    data_root = Path(__file__).parent / "data" / "patches" / "multi_scanner"
    model_dir = Path(__file__).parent / "models"
    output_dir = Path(__file__).parent / "ood_analysis_results"
    
    # Run analysis
    generate_ood_analysis(data_root, model_dir, output_dir)
