# %%
"""
SHAP Beeswarm Plot Visualization
================================
Generates SHAP beeswarm plots for each algorithm and outcome.
Shows feature importance and direction of impact.
Uses all transfusion timeframes (24h, 48h, 72h) as features.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
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
    'HistGradientBoosting': 'Histogram Gradient Boosting',
    'CatBoost': 'CatBoost',
    'ANN': 'MLP'
}

# Paths
models_dir = Path("../../result/models")
output_dir = Path("../../result/figures/shap")
output_dir.mkdir(parents=True, exist_ok=True)

# Number of top features to show
MAX_DISPLAY_FEATURES = 10


def save_figure_multi_format(fig, output_path, dpi=500):
    """Save figure in PNG, SVG, and PDF formats."""
    output_path = Path(output_path)
    base_path = output_path.with_suffix('')
    
    fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(f"{base_path}.svg", bbox_inches='tight', facecolor='white')
    fig.savefig(f"{base_path}.pdf", bbox_inches='tight', facecolor='white')
# Number of samples for SHAP calculation (for speed)
SHAP_SAMPLE_SIZE = 500
# Exclude missing indicator columns from display
EXCLUDE_MISSING_INDICATORS = True

# %%
def load_data_and_encoders():
    """Load train/test data and label encoders."""
    train_df = pd.read_parquet("../../data/processed/train_data.parquet")
    test_df = pd.read_parquet("../../data/processed/test_data.parquet")
    
    # Load label encoders
    label_encoders_path = models_dir / "label_encoders.pkl"
    if label_encoders_path.exists():
        with open(label_encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        for col, le in label_encoders.items():
            if col in train_df.columns:
                train_df[col] = le.transform(train_df[col].astype(str))
            if col in test_df.columns:
                test_df[col] = le.transform(test_df[col].astype(str))
    
    return train_df, test_df


def get_feature_cols(df):
    """Get feature columns using all transfusion timeframes (24h, 48h, 72h)."""
    exclude_cols = ID_COLS + TARGET_COLS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def filter_missing_indicators(shap_values, X_data, feature_names):
    """
    Filter out missing indicator columns from SHAP values and data.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        X_data: Feature data array (n_samples, n_features)
        feature_names: List of feature names
        
    Returns:
        Filtered shap_values, X_data, feature_names
    """
    if not EXCLUDE_MISSING_INDICATORS:
        return shap_values, X_data, feature_names
    
    # Find indices of non-missing indicator columns
    keep_indices = [i for i, name in enumerate(feature_names) if not name.endswith('_missing')]
    
    # Filter
    filtered_shap = shap_values[:, keep_indices]
    filtered_X = X_data[:, keep_indices]
    filtered_names = [feature_names[i] for i in keep_indices]
    
    return filtered_shap, filtered_X, filtered_names


# %%
def get_shap_explainer(model, model_name, X_background):
    """
    Create appropriate SHAP explainer based on model type.
    
    Args:
        model: Trained model
        model_name: Name of the model
        X_background: Background data for explainer
        
    Returns:
        SHAP explainer object
    """
    if model_name == 'DecisionTree':
        return shap.TreeExplainer(model)
    elif model_name in ['RandomForest', 'HistGradientBoosting', 'CatBoost']:
        return shap.TreeExplainer(model)
    elif model_name == 'ANN':
        # For ANN (Pipeline with scaler + MLP), use KernelExplainer
        # Sample background for speed
        if len(X_background) > 100:
            X_bg_sample = shap.sample(X_background, 100)
        else:
            X_bg_sample = X_background
        return shap.KernelExplainer(model.predict_proba, X_bg_sample)
    else:
        # Fallback to KernelExplainer
        if len(X_background) > 100:
            X_bg_sample = shap.sample(X_background, 100)
        else:
            X_bg_sample = X_background
        return shap.KernelExplainer(model.predict_proba, X_bg_sample)


def compute_shap_values(explainer, model_name, X_explain):
    """
    Compute SHAP values based on model type.
    
    Args:
        explainer: SHAP explainer
        model_name: Name of the model
        X_explain: Data to explain
        
    Returns:
        SHAP values for the positive class (2D array: n_samples x n_features)
    """
    if model_name == 'ANN':
        # KernelExplainer returns values for each class
        shap_values = explainer.shap_values(X_explain, nsamples=100)
        # Return values for positive class (index 1)
        if isinstance(shap_values, list):
            return shap_values[1]
        # If 3D array (samples, features, classes), take positive class
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            return shap_values[:, :, 1]
        return shap_values
    else:
        # TreeExplainer
        shap_values = explainer.shap_values(X_explain)
        
        # Handle different return formats
        if isinstance(shap_values, list):
            # For binary classification, return positive class
            return shap_values[1]
        elif hasattr(shap_values, 'values'):
            # shap.Explanation object
            vals = shap_values.values
            # If 3D (samples, features, classes), take positive class
            if isinstance(vals, np.ndarray) and vals.ndim == 3:
                return vals[:, :, 1]
            return vals
        elif isinstance(shap_values, np.ndarray):
            # If 3D array (samples, features, classes), take positive class
            if shap_values.ndim == 3:
                return shap_values[:, :, 1]
            return shap_values
        return shap_values


# %%
def get_top_features(shap_values, X_data, feature_names, top_n=10):
    """
    Get top N features by mean absolute SHAP value.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        X_data: Feature data array (n_samples, n_features)
        feature_names: List of feature names
        top_n: Number of top features to keep
        
    Returns:
        Filtered shap_values, X_data, feature_names for top N features only
    """
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Get indices of top N features
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    
    # Filter to top features only
    filtered_shap = shap_values[:, top_indices]
    filtered_X = X_data[:, top_indices]
    filtered_names = [feature_names[i] for i in top_indices]
    
    return filtered_shap, filtered_X, filtered_names


def create_beeswarm_plot(shap_values, X_explain, feature_names, title, output_path, max_display=10):
    """
    Create and save a SHAP beeswarm plot.
    
    Args:
        shap_values: SHAP values array
        X_explain: Feature values used for coloring
        feature_names: List of feature names
        title: Plot title
        output_path: Path to save the figure
        max_display: Maximum number of features to display
    """
    # Pre-filter to top N features to avoid "Sum of other features" row
    top_shap, top_X, top_names = get_top_features(
        shap_values, X_explain, feature_names, top_n=max_display
    )
    
    plt.figure(figsize=(12, 8))
    
    # Create SHAP Explanation object with only top features
    explanation = shap.Explanation(
        values=top_shap,
        data=top_X,
        feature_names=top_names
    )
    
    # Create beeswarm plot - show all features since we pre-filtered
    shap.plots.beeswarm(
        explanation, 
        max_display=max_display + 1,  # +1 to ensure no "sum of others" row
        show=False,
        plot_size=(12, 8)
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig = plt.gcf()
    save_figure_multi_format(fig, output_path)
    plt.close()
    
    base_name = Path(output_path).stem
    print(f"    Saved: {base_name}.png/svg/pdf")


# %%
def create_shap_summary_grid(shap_data_dict, target, output_path, max_display=10):
    """
    Create a grid of SHAP summary plots for all models.
    
    Args:
        shap_data_dict: Dictionary of {model_name: (shap_values, X_explain, feature_names)}
        target: Target name
        output_path: Path to save the figure
        max_display: Maximum features per subplot
    """
    n_models = len(shap_data_dict)
    if n_models == 0:
        return
    
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(7 * n_cols, 7 * n_rows))
    
    for idx, (model_name, (shap_values, X_explain, feature_names)) in enumerate(shap_data_dict.items()):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        
        # Pre-filter to top N features to avoid "Sum of other features" row
        top_shap, top_X, top_names = get_top_features(
            shap_values, X_explain, feature_names, top_n=max_display
        )
        
        # Create SHAP Explanation object with only top features
        explanation = shap.Explanation(
            values=top_shap,
            data=top_X,
            feature_names=top_names
        )
        
        plt.sca(ax)
        shap.plots.beeswarm(
            explanation,
            max_display=max_display + 1,  # +1 to ensure no "sum of others" row
            show=False,
            plot_size=None
        )
        
        model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        ax.set_title(model_display, fontsize=12, fontweight='bold')
    
    target_display = TARGET_DISPLAY_NAMES.get(target, target)
    fig.suptitle(f'SHAP Feature Importance - {target_display}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    save_figure_multi_format(fig, output_path)
    plt.close()
    
    base_name = Path(output_path).stem
    print(f"  Saved grid: {base_name}.png/svg/pdf")


# %%
# Load data
print("Loading data...")
train_df, test_df = load_data_and_encoders()
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Use test set only for SHAP analysis
# Get feature columns (all timeframes)
feature_cols = get_feature_cols(test_df)
print(f"Number of features: {len(feature_cols)}")

# Sample data for SHAP (for speed)
sample_size = min(SHAP_SAMPLE_SIZE, len(test_df))
sample_df = test_df.sample(n=sample_size, random_state=42)
X_sample = sample_df[feature_cols].values

# Background data (smaller sample)
bg_size = min(100, len(train_df))
X_background = train_df[feature_cols].sample(n=bg_size, random_state=42).values

# %%
# Generate SHAP plots for each target and model
print("\n" + "="*60)
print("Generating SHAP beeswarm plots...")
print("="*60)

for target in TARGET_COLS:
    print(f"\n  Target: {target}")
    
    shap_data_dict = {}  # Store SHAP data for grid plot
    
    for model_name in MODEL_NAMES:
        model_path = models_dir / f"{target}_{model_name}.pkl"
        
        if not model_path.exists():
            print(f"    {model_name}: Model not found, skipping...")
            continue
        
        print(f"    {model_name}: Computing SHAP values...")
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Create explainer
            explainer = get_shap_explainer(model, model_name, X_background)
            
            # Compute SHAP values
            shap_values = compute_shap_values(explainer, model_name, X_sample)
            
            # Filter out missing indicators
            filtered_shap, filtered_X, filtered_names = filter_missing_indicators(
                shap_values, X_sample, feature_cols
            )
            
            # Store for grid plot
            shap_data_dict[model_name] = (filtered_shap, filtered_X, filtered_names)
            
            # Create individual beeswarm plot
            target_display = TARGET_DISPLAY_NAMES.get(target, target)
            model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            title = f'{model_display}\n{target_display}'
            
            output_path = output_dir / f"{target}_{model_name}_shap.png"
            create_beeswarm_plot(
                filtered_shap, filtered_X, filtered_names,
                title, output_path, max_display=MAX_DISPLAY_FEATURES
            )
            
        except Exception as e:
            print(f"    {model_name}: Error - {str(e)}")
            continue
    
    # Create grid plot for all models
    if shap_data_dict:
        grid_output_path = output_dir / f"{target}_all_models_shap.png"
        create_shap_summary_grid(shap_data_dict, target, grid_output_path)

# %%
print("\n" + "="*60)
print("SHAP VISUALIZATION COMPLETE!")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print(f"Individual plots: {{target}}_{{model}}_shap.png")
print(f"Grid plots: {{target}}_all_models_shap.png")


# %%
