# %%
"""
Model Performance Table - TRANSFUSED PATIENTS ONLY
===================================================
Creates comprehensive performance tables with classification metrics
for the transfused patients subgroup.
Uses all transfusion timeframes (24h, 48h, 72h) as features.

Binary metrics are calculated at the optimal Youden index threshold.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

warnings.filterwarnings('ignore')

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
TRANSFUSION_TYPES = ['rbc_volume', 'ffp_volume', 'platelet_volume']
TIMEFRAMES = ['24h', '48h', '72h']
ID_COLS = ['subject_id', 'hadm_id']

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
models_dir = Path("../../result/models_transfused")
output_dir = Path("../../result/tables_transfused")
output_dir.mkdir(parents=True, exist_ok=True)


# %%
def has_any_transfusion(df):
    """Check if patient received any transfusion within any timeframe."""
    transfusion_cols = []
    for tf in TIMEFRAMES:
        for t in TRANSFUSION_TYPES:
            col = f"{t}_{tf}"
            if col in df.columns:
                transfusion_cols.append(col)
    return (df[transfusion_cols].fillna(0) > 0).any(axis=1)


def filter_transfused_patients(df):
    """Filter to patients who received any transfusion."""
    mask = has_any_transfusion(df)
    return df[mask].copy()


def get_feature_cols(df):
    """Get feature columns using all transfusion timeframes (24h, 48h, 72h)."""
    exclude_cols = ID_COLS + TARGET_COLS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def find_youden_threshold(y_true, y_pred_proba):
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def calculate_metrics_at_threshold(y_true, y_pred_proba, threshold):
    """Calculate all classification metrics at a given threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    
    return {
        'Threshold': threshold,
        'AUROC': auroc,
        'AUPRC': auprc,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1': f1,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }


def bootstrap_ci(y_true, y_pred_proba, threshold, n_bootstrap=1000, ci=0.95, random_state=42):
    """
    Calculate bootstrap confidence intervals for all metrics.
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    metrics_names = ['AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']
    bootstrap_results = {m: [] for m in metrics_names}
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_pred_proba_boot = y_pred_proba[indices]
        
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        metrics = calculate_metrics_at_threshold(y_true_boot, y_pred_proba_boot, threshold)
        
        for m in metrics_names:
            bootstrap_results[m].append(metrics[m])
    
    alpha = 1 - ci
    ci_results = {}
    
    for m in metrics_names:
        values = np.array(bootstrap_results[m])
        if len(values) > 0:
            ci_results[f'{m}_lower'] = np.percentile(values, alpha/2 * 100)
            ci_results[f'{m}_upper'] = np.percentile(values, (1 - alpha/2) * 100)
        else:
            ci_results[f'{m}_lower'] = np.nan
            ci_results[f'{m}_upper'] = np.nan
    
    return ci_results


# %%
def load_data_and_encoders():
    """Load test data and label encoders."""
    test_df = pd.read_parquet("../../data/processed/test_data.parquet")
    
    label_encoders_path = models_dir / "label_encoders.pkl"
    if label_encoders_path.exists():
        with open(label_encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        for col, le in label_encoders.items():
            if col in test_df.columns:
                test_df[col] = le.transform(test_df[col].astype(str))
    
    return test_df


# %%
# Check if transfused models exist
if not models_dir.exists():
    print("ERROR: Models not found. Please run sklearn_train_transfused.py first.")
    exit(1)

# Load test data
print("Loading test data...")
test_df_orig = load_data_and_encoders()
print(f"Original test data shape: {test_df_orig.shape}")

# Filter to transfused patients
test_df = filter_transfused_patients(test_df_orig)
print(f"Transfused patients in test set: {len(test_df)}")

# Get feature columns (all timeframes)
feature_cols = get_feature_cols(test_df)
X_test = test_df[feature_cols].values
print(f"Number of features: {len(feature_cols)}")

# %%
# Calculate metrics for all models
print("\n" + "="*60)
print("Calculating performance metrics with Bootstrap CI (TRANSFUSED PATIENTS)")
print("="*60)

all_results = []

for target in TARGET_COLS:
    print(f"\nTarget: {target}")
    y_test = test_df[target].values
    
    for model_name in MODEL_NAMES:
        model_path = models_dir / f"{target}_{model_name}.pkl"
        
        if not model_path.exists():
            print(f"  {model_name}: Model not found, skipping...")
            continue
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        optimal_threshold = find_youden_threshold(y_test, y_pred_proba)
        metrics = calculate_metrics_at_threshold(y_test, y_pred_proba, optimal_threshold)
        
        # Calculate bootstrap CI
        ci_metrics = bootstrap_ci(y_test, y_pred_proba, optimal_threshold, n_bootstrap=1000)
        metrics.update(ci_metrics)
        
        metrics['Target'] = TARGET_DISPLAY_NAMES.get(target, target)
        metrics['Target_Raw'] = target
        metrics['Model'] = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        metrics['Model_Raw'] = model_name
        metrics['N_Test'] = len(test_df)
        
        all_results.append(metrics)
        
        print(f"  {model_name}: AUROC={metrics['AUROC']:.3f} ({metrics['AUROC_lower']:.3f}-{metrics['AUROC_upper']:.3f})")

# %%
# Create results DataFrame
results_df = pd.DataFrame(all_results)

col_order = [
    'Target', 'Model', 'N_Test',
    'AUROC', 'AUROC_lower', 'AUROC_upper',
    'AUPRC', 'AUPRC_lower', 'AUPRC_upper',
    'Threshold',
    'Accuracy', 'Accuracy_lower', 'Accuracy_upper',
    'Sensitivity', 'Sensitivity_lower', 'Sensitivity_upper',
    'Specificity', 'Specificity_lower', 'Specificity_upper',
    'PPV', 'PPV_lower', 'PPV_upper',
    'NPV', 'NPV_lower', 'NPV_upper',
    'F1', 'F1_lower', 'F1_upper',
    'TP', 'TN', 'FP', 'FN',
    'Target_Raw', 'Model_Raw'
]
results_df = results_df[[c for c in col_order if c in results_df.columns]]

numeric_cols = ['AUROC', 'AUROC_lower', 'AUROC_upper',
                'AUPRC', 'AUPRC_lower', 'AUPRC_upper', 'Threshold',
                'Accuracy', 'Accuracy_lower', 'Accuracy_upper',
                'Sensitivity', 'Sensitivity_lower', 'Sensitivity_upper',
                'Specificity', 'Specificity_lower', 'Specificity_upper',
                'PPV', 'PPV_lower', 'PPV_upper',
                'NPV', 'NPV_lower', 'NPV_upper',
                'F1', 'F1_lower', 'F1_upper']
for col in numeric_cols:
    if col in results_df.columns:
        results_df[col] = results_df[col].round(3)

# Create formatted columns with CI
for metric in ['AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']:
    if f'{metric}_lower' in results_df.columns:
        results_df[f'{metric}_CI'] = results_df.apply(
            lambda row: f"{row[metric]:.3f} ({row[f'{metric}_lower']:.3f}-{row[f'{metric}_upper']:.3f})", axis=1)

print("\n" + "="*60)
print("Results Preview (TRANSFUSED PATIENTS):")
print("="*60)
print(results_df[['Target', 'Model', 'N_Test', 'AUROC_CI', 'AUPRC_CI', 
                   'Sensitivity_CI', 'Specificity_CI']].to_string(index=False))

# %%
# Save results
full_results_path = output_dir / "model_performance_full_transfused.csv"
results_df.to_csv(full_results_path, index=False)
print(f"\nFull results saved: {full_results_path}")

# Summary with CI
summary_cols_ci = ['Target', 'Model', 'N_Test', 'AUROC_CI', 'AUPRC_CI', 'Threshold',
                   'Accuracy_CI', 'Sensitivity_CI', 'Specificity_CI', 'PPV_CI', 'NPV_CI', 'F1_CI']
summary_df_ci = results_df[[c for c in summary_cols_ci if c in results_df.columns]].copy()
summary_df_ci.columns = [c.replace('_CI', '') for c in summary_df_ci.columns]

summary_path = output_dir / "model_performance_summary_transfused.csv"
summary_df_ci.to_csv(summary_path, index=False)
print(f"Summary saved: {summary_path}")

# Raw values
summary_cols_raw = ['Target', 'Model', 'N_Test', 'AUROC', 'AUROC_lower', 'AUROC_upper',
                    'AUPRC', 'AUPRC_lower', 'AUPRC_upper', 'Threshold',
                    'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']
summary_df_raw = results_df[[c for c in summary_cols_raw if c in results_df.columns]].copy()
raw_path = output_dir / "model_performance_raw_transfused.csv"
summary_df_raw.to_csv(raw_path, index=False)
print(f"Raw values saved: {raw_path}")

# %%
# Create Excel file
excel_path = output_dir / "model_performance_transfused.xlsx"

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Summary with CI
    summary_df_ci.to_excel(writer, sheet_name='All Results', index=False)
    
    # Raw values
    summary_df_raw.to_excel(writer, sheet_name='Raw Values', index=False)
    
    # Per target sheets
    for target in TARGET_COLS:
        target_display = TARGET_DISPLAY_NAMES.get(target, target)
        target_df = summary_df_ci[summary_df_ci['Target'] == target_display].copy()
        
        # Truncate sheet name if too long (Excel limit is 31 chars)
        sheet_name = target[:28] if len(target) > 28 else target
        target_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Excel file saved: {excel_path}")

# %%
# Compare with full cohort results (use raw values file)
print("\n" + "="*60)
print("Comparison: Full Cohort vs Transfused Subgroup")
print("="*60)

full_cohort_path = Path("../../result/tables/model_performance_raw.csv")
if full_cohort_path.exists():
    full_df = pd.read_csv(full_cohort_path)
    
    comparison_rows = []
    for _, row in results_df.iterrows():
        full_match = full_df[
            (full_df['Target'] == row['Target']) &
            (full_df['Model'] == row['Model'])
        ]
        
        if len(full_match) > 0:
            comparison_rows.append({
                'Target': row['Target'],
                'Model': row['Model'],
                'Full Cohort N': full_match.iloc[0].get('N_Test', 'N/A'),
                'Transfused N': row['N_Test'],
                'Full AUROC': full_match.iloc[0]['AUROC'],
                'Transfused AUROC': row['AUROC'],
                'AUROC Diff': row['AUROC'] - full_match.iloc[0]['AUROC'],
                'Full AUPRC': full_match.iloc[0]['AUPRC'],
                'Transfused AUPRC': row['AUPRC'],
                'AUPRC Diff': row['AUPRC'] - full_match.iloc[0]['AUPRC'],
            })
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Round
    for col in ['Full AUROC', 'Transfused AUROC', 'AUROC Diff', 
                'Full AUPRC', 'Transfused AUPRC', 'AUPRC Diff']:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].round(3)
    
    comparison_path = output_dir / "comparison_full_vs_transfused.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison saved: {comparison_path}")
    
    # Summary pivot
    print("\nMean AUROC Difference (Transfused - Full) by Model:")
    pivot = comparison_df.groupby('Model')['AUROC Diff'].mean().round(3)
    print(pivot.to_string())

# %%
# Best models summary
print("\n" + "="*60)
print("Best Model per Target (TRANSFUSED PATIENTS)")
print("="*60)

best_models = []
for target in TARGET_COLS:
    target_display = TARGET_DISPLAY_NAMES.get(target, target)
    subset = results_df[results_df['Target'] == target_display]
    if len(subset) > 0:
        best_idx = subset['AUROC'].idxmax()
        best_row = subset.loc[best_idx]
        best_models.append({
            'Target': target_display,
            'Best Model': best_row['Model'],
            'N_Test': best_row['N_Test'],
            'AUROC': best_row['AUROC_CI'] if 'AUROC_CI' in best_row else best_row['AUROC'],
            'AUPRC': best_row['AUPRC_CI'] if 'AUPRC_CI' in best_row else best_row['AUPRC'],
            'Sensitivity': best_row['Sensitivity_CI'] if 'Sensitivity_CI' in best_row else best_row['Sensitivity'],
            'Specificity': best_row['Specificity_CI'] if 'Specificity_CI' in best_row else best_row['Specificity']
        })

best_models_df = pd.DataFrame(best_models)
print(best_models_df.to_string(index=False))

best_models_path = output_dir / "best_models_summary_transfused.csv"
best_models_df.to_csv(best_models_path, index=False)

# %%
print("\n" + "="*60)
print("TABLE GENERATION COMPLETE (TRANSFUSED PATIENTS)")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print(f"\nGenerated files:")
print(f"  - model_performance_full_transfused.csv")
print(f"  - model_performance_summary_transfused.csv")
print(f"  - model_performance_transfused.xlsx")
print(f"  - comparison_full_vs_transfused.csv")
print(f"  - best_models_summary_transfused.csv")


# %%
