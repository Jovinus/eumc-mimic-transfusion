# %%
"""
ROC and Precision-Recall Curve Visualization - ABLATION STUDY (No Medical Scores)
==================================================================================
Generates combined AUROC and AUPRC plots for each algorithm and outcome
for the ablation study excluding medical scores (SOFA, APACHE2).
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, roc_auc_score

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# %%
# Configuration
TARGET_COLS = [
    'icu_mortality',
    'hospital_mortality', 
    'mortality_28day',
    'blood_culture_positive_after_72h',
    'new_antibiotic_after_72h'
]
MODEL_NAMES = ['DecisionTree', 'RandomForest', 'HistGradientBoosting', 'CatBoost', 'ANN']
ID_COLS = ['subject_id', 'hadm_id']

MODEL_COLORS = {
    'DecisionTree': '#E74C3C',
    'RandomForest': '#27AE60',
    'HistGradientBoosting': '#3498DB',
    'CatBoost': '#9B59B6',
    'ANN': '#F39C12'
}

TARGET_DISPLAY_NAMES = {
    'icu_mortality': 'ICU Mortality',
    'hospital_mortality': 'Hospital Mortality',
    'mortality_28day': '28-Day Mortality',
    'blood_culture_positive_after_72h': 'Blood Culture Positive (>72h)',
    'new_antibiotic_after_72h': 'New Antibiotic (>72h)'
}

MODEL_DISPLAY_NAMES = {
    'DecisionTree': 'Decision Tree',
    'RandomForest': 'Random Forest',
    'HistGradientBoosting': 'Hist Gradient Boosting',
    'CatBoost': 'CatBoost',
    'ANN': 'MLP'
}

# Paths
models_dir = Path("../../result/models_ablation_no_scores")
output_dir = Path("../../result/figures_ablation_no_scores/curves")
output_dir.mkdir(parents=True, exist_ok=True)


def save_figure_multi_format(fig, output_path, dpi=500):
    """Save figure in PNG, SVG, and PDF formats."""
    output_path = Path(output_path)
    base_path = output_path.with_suffix('')
    
    fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(f"{base_path}.svg", bbox_inches='tight', facecolor='white')
    fig.savefig(f"{base_path}.pdf", bbox_inches='tight', facecolor='white')


# %%
def get_feature_cols_no_scores(df, excluded_cols):
    """Get feature columns EXCLUDING medical scores."""
    exclude_cols = ID_COLS + TARGET_COLS + excluded_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def load_test_data():
    """Load test data and apply encoding."""
    test_df = pd.read_parquet("../../data/processed/test_data.parquet")
    
    label_encoders_path = models_dir / "label_encoders.pkl"
    if label_encoders_path.exists():
        with open(label_encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        for col, le in label_encoders.items():
            if col in test_df.columns:
                test_df[col] = le.transform(test_df[col].astype(str))
    
    return test_df


def load_excluded_cols():
    """Load the list of excluded columns."""
    excluded_path = models_dir / "excluded_columns.pkl"
    if excluded_path.exists():
        with open(excluded_path, 'rb') as f:
            return pickle.load(f)
    return []


# %%
def plot_roc_prc_single_target(test_df, target, models_dict, feature_cols, output_path):
    """Create a combined ROC and PRC plot for a single target."""
    X_test = test_df[feature_cols].values
    y_test = test_df[target].values
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random (AUC = 0.50)')
    
    ax_prc = axes[1]
    baseline = y_test.mean()
    ax_prc.axhline(y=baseline, color='k', linestyle='--', lw=1.5, alpha=0.7, 
                   label=f'Random (AP = {baseline:.2f})')
    
    for model_name, model in models_dict.items():
        color = MODEL_COLORS.get(model_name, '#333333')
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        ax_roc.plot(fpr, tpr, color=color, lw=2.5, 
                    label=f'{model_display} (AUC = {roc_auc:.3f})')
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        ax_prc.plot(recall, precision, color=color, lw=2.5,
                    label=f'{model_display} (AP = {ap:.3f})')
    
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right', fontsize=8, framealpha=0.95)
    
    ax_prc.set_xlim([-0.02, 1.02])
    ax_prc.set_ylim([-0.02, 1.02])
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel('Precision')
    ax_prc.set_title('Precision-Recall Curve')
    ax_prc.legend(loc='upper right', fontsize=8, framealpha=0.95)
    
    target_display = TARGET_DISPLAY_NAMES.get(target, target)
    n_patients = len(test_df)
    fig.suptitle(f'{target_display} - Ablation: No Medical Scores', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    save_figure_multi_format(fig, output_path)
    plt.close()
    
    base_name = Path(output_path).stem
    print(f"  Saved: {base_name}.png/svg/pdf")


# %%
def plot_all_models_comparison(test_df, feature_cols, output_path):
    """Create a grid of ROC and PRC curves for all targets."""
    n_targets = len(TARGET_COLS)
    fig, axes = plt.subplots(n_targets, 2, figsize=(10, 4.5 * n_targets))
    
    X_test = test_df[feature_cols].values
    
    for idx, target in enumerate(TARGET_COLS):
        y_test = test_df[target].values
        
        ax_roc = axes[idx, 0]
        ax_prc = axes[idx, 1]
        
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random')
        baseline = y_test.mean()
        ax_prc.axhline(y=baseline, color='k', linestyle='--', lw=1.5, alpha=0.7, label='Random')
        
        for model_name in MODEL_NAMES:
            model_path = models_dir / f"{target}_{model_name}.pkl"
            if not model_path.exists():
                continue
                
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            color = MODEL_COLORS.get(model_name, '#333333')
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{model_display} ({roc_auc:.3f})')
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)
            ax_prc.plot(recall, precision, color=color, lw=2, label=f'{model_display} ({ap:.3f})')
        
        target_display = TARGET_DISPLAY_NAMES.get(target, target)
        ax_roc.set_title(f'{target_display} - ROC')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc='lower right', fontsize=7)
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.02])
        
        ax_prc.set_title(f'{target_display} - PRC')
        ax_prc.set_xlabel('Recall')
        ax_prc.set_ylabel('Precision')
        ax_prc.legend(loc='upper right', fontsize=7)
        ax_prc.set_xlim([-0.02, 1.02])
        ax_prc.set_ylim([-0.02, 1.02])
    
    n_patients = len(test_df)
    fig.suptitle(f'Model Performance - Ablation: No Medical Scores', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.3)
    save_figure_multi_format(fig, output_path)
    plt.close()
    
    base_name = Path(output_path).stem
    print(f"Saved comparison grid: {base_name}.png/svg/pdf")


# %%
def plot_comparison_with_full_model(test_df, feature_cols, excluded_cols, output_path):
    """Create comparison plots between full model and ablation model."""
    full_models_dir = Path("../../result/models")
    if not full_models_dir.exists():
        print("Full model directory not found. Skipping comparison plot.")
        return
    
    # Load full model feature columns
    full_feature_cols_path = full_models_dir / "feature_columns.pkl"
    if not full_feature_cols_path.exists():
        print("Full model feature columns not found. Skipping comparison plot.")
        return
    
    with open(full_feature_cols_path, 'rb') as f:
        full_feature_cols = pickle.load(f)
    
    n_targets = len(TARGET_COLS)
    fig, axes = plt.subplots(n_targets, 2, figsize=(10, 4.5 * n_targets))
    
    for idx, target in enumerate(TARGET_COLS):
        y_test = test_df[target].values
        
        ax_roc = axes[idx, 0]
        ax_prc = axes[idx, 1]
        
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random')
        baseline = y_test.mean()
        ax_prc.axhline(y=baseline, color='k', linestyle='--', lw=1.5, alpha=0.7, label='Random')
        
        # Plot best model from full and ablation
        for model_type, mdir, linestyle, suffix in [
            ('Full', full_models_dir, '-', ''),
            ('No Scores', models_dir, '--', ' (Ablation)')
        ]:
            best_info_path = mdir / "best" / "best_model_info.pkl"
            if not best_info_path.exists():
                continue
            
            with open(best_info_path, 'rb') as f:
                best_info = pickle.load(f)
            
            if target not in best_info:
                continue
            
            best_model_name = best_info[target]['model_name']
            model_path = mdir / f"{target}_{best_model_name}.pkl"
            
            if not model_path.exists():
                continue
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Use appropriate feature columns
            if model_type == 'Full':
                X_test = test_df[full_feature_cols].values
            else:
                X_test = test_df[feature_cols].values
            
            color = MODEL_COLORS.get(best_model_name, '#333333')
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=color, lw=2.5, linestyle=linestyle,
                       label=f'{model_type}: {best_model_name} ({roc_auc:.3f})')
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)
            ax_prc.plot(recall, precision, color=color, lw=2.5, linestyle=linestyle,
                       label=f'{model_type}: {best_model_name} ({ap:.3f})')
        
        target_display = TARGET_DISPLAY_NAMES.get(target, target)
        ax_roc.set_title(f'{target_display} - ROC')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc='lower right', fontsize=9)
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.02])
        
        ax_prc.set_title(f'{target_display} - PRC')
        ax_prc.set_xlabel('Recall')
        ax_prc.set_ylabel('Precision')
        ax_prc.legend(loc='upper right', fontsize=9)
        ax_prc.set_xlim([-0.02, 1.02])
        ax_prc.set_ylim([-0.02, 1.02])
    
    fig.suptitle('Best Model Comparison: Full vs Ablation (No Medical Scores)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.3)
    save_figure_multi_format(fig, output_path)
    plt.close()
    
    base_name = Path(output_path).stem
    print(f"Saved comparison plot: {base_name}.png/svg/pdf")


# %%
# Check if models exist
if not models_dir.exists():
    print("ERROR: Ablation models not found. Please run sklearn_train_no_scores.py first.")
    exit(1)

# Load test data
print("Loading test data...")
test_df = load_test_data()
print(f"Test data shape: {test_df.shape}")

# Load excluded columns
excluded_cols = load_excluded_cols()
print(f"Excluded columns: {excluded_cols}")

# Get feature columns
feature_cols = get_feature_cols_no_scores(test_df, excluded_cols)
print(f"Number of features: {len(feature_cols)}")

# %%
# Generate plots
print("\n" + "="*60)
print("Generating ROC and PRC curves (ABLATION: NO MEDICAL SCORES)")
print("="*60)

for target in TARGET_COLS:
    print(f"  Target: {target}")
    
    models_dict = {}
    for model_name in MODEL_NAMES:
        model_path = models_dir / f"{target}_{model_name}.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models_dict[model_name] = pickle.load(f)
    
    if models_dict:
        output_path = output_dir / f"{target}_roc_prc_ablation_no_scores.png"
        plot_roc_prc_single_target(test_df, target, models_dict, feature_cols, output_path)

# %%
# Generate comparison grid
print("\n" + "="*60)
print("Generating comparison grid...")
print("="*60)

output_path = output_dir / "all_targets_comparison_ablation_no_scores.png"
plot_all_models_comparison(test_df, feature_cols, output_path)

# %%
# Generate comparison with full model
print("\n" + "="*60)
print("Generating full vs ablation comparison...")
print("="*60)

output_path = output_dir / "full_vs_ablation_comparison.png"
plot_comparison_with_full_model(test_df, feature_cols, excluded_cols, output_path)

# %%
print("\n" + "="*60)
print("VISUALIZATION COMPLETE (ABLATION: NO MEDICAL SCORES)")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print(f"Individual plots: {len(TARGET_COLS)} files")
print(f"Comparison grid: 1 file")
print(f"Full vs Ablation comparison: 1 file")
