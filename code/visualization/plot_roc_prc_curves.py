# %%
"""
ROC and Precision-Recall Curve Visualization
=============================================
Generates combined AUROC and AUPRC plots for each algorithm and outcome.
Uses all transfusion timeframes (24h, 48h, 72h) as features.
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

# Color palette for models
MODEL_COLORS = {
    'DecisionTree': '#E74C3C',
    'RandomForest': '#27AE60',
    'HistGradientBoosting': '#3498DB',
    'CatBoost': '#9B59B6',
    'ANN': '#F39C12'
}

# Nice target names for display
TARGET_DISPLAY_NAMES = {
    'icu_mortality': 'ICU Mortality',
    'hospital_mortality': 'Hospital Mortality',
    'mortality_28day': '28-Day Mortality',
    'blood_culture_positive_after_72h': 'Blood Culture Positive (>72h)',
    'new_antibiotic_after_72h': 'New Antibiotic (>72h)'
}

# Model display names
MODEL_DISPLAY_NAMES = {
    'DecisionTree': 'Decision Tree',
    'RandomForest': 'Random Forest',
    'HistGradientBoosting': 'Hist Gradient Boosting',
    'CatBoost': 'CatBoost',
    'ANN': 'MLP'
}

# Paths
models_dir = Path("../../result/models")
predictions_dir = Path("../../result/predictions")
output_dir = Path("../../result/figures/curves")
output_dir.mkdir(parents=True, exist_ok=True)


def save_figure_multi_format(fig, output_path, dpi=500):
    """Save figure in PNG, SVG, and PDF formats."""
    # Ensure output_path is a Path object and has .png extension
    output_path = Path(output_path)
    base_path = output_path.with_suffix('')
    
    # Save in all formats
    fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(f"{base_path}.svg", bbox_inches='tight', facecolor='white')
    fig.savefig(f"{base_path}.pdf", bbox_inches='tight', facecolor='white')

# %%
def load_test_data():
    """Load test data for evaluation with label encoding."""
    test_df = pd.read_parquet("../../data/processed/test_data.parquet")
    
    # Load and apply label encoders
    label_encoders_path = models_dir / "label_encoders.pkl"
    if label_encoders_path.exists():
        with open(label_encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        for col, le in label_encoders.items():
            if col in test_df.columns:
                test_df[col] = le.transform(test_df[col].astype(str))
    
    return test_df


def get_feature_cols(df):
    """Get feature columns using all transfusion timeframes (24h, 48h, 72h)."""
    exclude_cols = ID_COLS + TARGET_COLS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


# %%
def plot_roc_prc_single_target(test_df, target, models_dict, feature_cols, output_path):
    """
    Create a combined ROC and PRC plot for a single target.
    
    Args:
        test_df: Test dataframe
        target: Target variable name
        models_dict: Dictionary of {model_name: model}
        feature_cols: List of feature column names
        output_path: Path to save the figure
    """
    X_test = test_df[feature_cols].values
    y_test = test_df[target].values
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # ROC Curve subplot
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random (AUC = 0.50)')
    
    # PRC subplot
    ax_prc = axes[1]
    baseline = y_test.mean()
    ax_prc.axhline(y=baseline, color='k', linestyle='--', lw=1.5, alpha=0.7, 
                   label=f'Random (AP = {baseline:.2f})')
    
    # Plot curves for each model
    for model_name, model in models_dict.items():
        color = MODEL_COLORS.get(model_name, '#333333')
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        ax_roc.plot(fpr, tpr, color=color, lw=2.5, 
                    label=f'{model_display} (AUC = {roc_auc:.3f})')
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        ax_prc.plot(recall, precision, color=color, lw=2.5,
                    label=f'{model_display} (AP = {ap:.3f})')
    
    # Format ROC subplot
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right', fontsize=8, framealpha=0.95)
    
    # Format PRC subplot
    ax_prc.set_xlim([-0.02, 1.02])
    ax_prc.set_ylim([-0.02, 1.02])
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel('Precision')
    ax_prc.set_title('Precision-Recall Curve')
    ax_prc.legend(loc='upper right', fontsize=8, framealpha=0.95)
    
    # Overall title
    target_display = TARGET_DISPLAY_NAMES.get(target, target)
    fig.suptitle(f'{target_display}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    save_figure_multi_format(fig, output_path)
    plt.close()
    
    base_name = Path(output_path).stem
    print(f"  Saved: {base_name}.png/svg/pdf")


# %%
def plot_all_models_comparison(test_df, feature_cols, output_path):
    """
    Create a grid of ROC and PRC curves for all targets.
    """
    n_targets = len(TARGET_COLS)
    fig, axes = plt.subplots(n_targets, 2, figsize=(10, 4.5 * n_targets))
    
    X_test = test_df[feature_cols].values
    
    for idx, target in enumerate(TARGET_COLS):
        y_test = test_df[target].values
        
        ax_roc = axes[idx, 0]
        ax_prc = axes[idx, 1]
        
        # Baseline lines
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random')
        baseline = y_test.mean()
        ax_prc.axhline(y=baseline, color='k', linestyle='--', lw=1.5, alpha=0.7, label='Random')
        
        # Load models and plot
        for model_name in MODEL_NAMES:
            model_path = models_dir / f"{target}_{model_name}.pkl"
            if not model_path.exists():
                continue
                
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            color = MODEL_COLORS.get(model_name, '#333333')
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{model_display} ({roc_auc:.3f})')
            
            # PRC
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)
            ax_prc.plot(recall, precision, color=color, lw=2, label=f'{model_display} ({ap:.3f})')
        
        # Format
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
    
    fig.suptitle('Model Performance Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.3)
    save_figure_multi_format(fig, output_path)
    plt.close()
    
    base_name = Path(output_path).stem
    print(f"Saved comparison grid: {base_name}.png/svg/pdf")


# %%
# Load test data
print("Loading test data...")
test_df = load_test_data()
print(f"Test data shape: {test_df.shape}")

# Get feature columns
feature_cols = get_feature_cols(test_df)
print(f"Number of features: {len(feature_cols)}")

# %%
# Generate individual plots for each target
print("\n" + "="*60)
print("Generating ROC and PRC curves...")
print("="*60)

for target in TARGET_COLS:
    print(f"  Target: {target}")
    
    # Load all models for this target
    models_dict = {}
    for model_name in MODEL_NAMES:
        model_path = models_dir / f"{target}_{model_name}.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models_dict[model_name] = pickle.load(f)
    
    if models_dict:
        output_path = output_dir / f"{target}_roc_prc.png"
        plot_roc_prc_single_target(test_df, target, models_dict, feature_cols, output_path)

# %%
# Generate comparison grid
print("\n" + "="*60)
print("Generating comparison grid...")
print("="*60)

output_path = output_dir / "all_targets_comparison.png"
plot_all_models_comparison(test_df, feature_cols, output_path)

# %%
print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print(f"Individual plots: {len(TARGET_COLS)} files")
print(f"Comparison grid: 1 file")


# %%
