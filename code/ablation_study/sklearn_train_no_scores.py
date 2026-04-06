# %%
"""
Classification Model Training - ABLATION STUDY (No Medical Scores)
===================================================================
Ablation study excluding medical severity scores (SOFA, APACHE2) from features.

This experiment evaluates model performance without clinical scoring systems
to understand their contribution to prediction accuracy.

Excluded features:
- sofa_resp, sofa_coag, sofa_liver, sofa_cardio, sofa_cns, sofa_renal, sofa_total
- apache2_score
- All related missing indicators (*_missing)

Algorithms: Decision Tree, Random Forest, HistGradientBoosting, CatBoost, ANN (MLP)
Hyperparameter tuning with 5-fold cross-validation
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# %%
# Configuration
RANDOM_STATE = 1004
CV_FOLDS = 5
N_JOBS = -1

TARGET_COLS = [
    'icu_mortality',
    'hospital_mortality', 
    'mortality_28day',
    'blood_culture_positive_after_72h',
    'new_antibiotic_after_72h'
]

ID_COLS = ['subject_id', 'hadm_id']
CATEGORICAL_COLS = ['surgery_type']

# Medical scores to EXCLUDE from ablation study
MEDICAL_SCORE_COLS = [
    'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 
    'sofa_cns', 'sofa_renal', 'sofa_total', 'apache2_score',
    'sofa_resp_missing', 'sofa_coag_missing', 'sofa_liver_missing',
    'sofa_cardio_missing', 'sofa_cns_missing', 'sofa_renal_missing',
    'sofa_total_missing', 'apache2_score_missing'
]

# %%
# Load processed data
print("Loading processed data...")
train_df = pd.read_parquet("../../data/processed/train_data.parquet")
test_df = pd.read_parquet("../../data/processed/test_data.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# %%
# Encode categorical columns
label_encoders = {}
for col in CATEGORICAL_COLS:
    if col in train_df.columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {le.classes_}")


# %%
def get_feature_cols_no_scores(df):
    """Get feature columns EXCLUDING medical scores (SOFA, APACHE2)."""
    exclude_cols = ID_COLS + TARGET_COLS + MEDICAL_SCORE_COLS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


# %%
# Get feature columns (excluding medical scores)
feature_cols = get_feature_cols_no_scores(train_df)
print(f"\n{'='*60}")
print("ABLATION STUDY: Excluding Medical Scores")
print(f"{'='*60}")
print(f"Excluded columns: {MEDICAL_SCORE_COLS}")
print(f"Number of features (after exclusion): {len(feature_cols)}")
print(f"\nFeature columns:")
for i, col in enumerate(feature_cols):
    print(f"  {i+1}. {col}")

# %%
# Define hyperparameter grids
param_grids = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'params': {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'criterion': ['gini', 'entropy']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
    },
    'HistGradientBoosting': {
        'model': HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        'params': {
            'learning_rate': [0.05, 0.1],
            'max_iter': [100, 200],
            'max_depth': [5, 7, 10],
            'min_samples_leaf': [20, 50],
            'l2_regularization': [0, 0.1]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(
            random_state=RANDOM_STATE, 
            verbose=0,
            thread_count=N_JOBS
        ),
        'params': {
            'learning_rate': [0.05, 0.1],
            'depth': [4, 6, 8],
            'iterations': [100, 200],
            'l2_leaf_reg': [1, 3, 5]
        }
    },
    'ANN': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(random_state=RANDOM_STATE, max_iter=300, early_stopping=True))
        ]),
        'params': {
            'mlp__hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
            'mlp__activation': ['relu'],
            'mlp__alpha': [0.001, 0.01],
            'mlp__learning_rate_init': [0.001, 0.01],
            'mlp__batch_size': [64, 128]
        }
    }
}


# %%
def train_and_evaluate(X_train, y_train, X_test, y_test, model_name, model, params, cv_folds=5):
    """Train model with GridSearchCV and evaluate on test set."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=cv,
        scoring='roc_auc',
        n_jobs=N_JOBS,
        verbose=1,
        refit=True
    )
    
    grid_search.fit(X_train, y_train)
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV ROC-AUC: {best_cv_score:.4f}")
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model_name': model_name,
        'best_params': best_params,
        'cv_roc_auc': best_cv_score,
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_pred, zero_division=0),
        'test_average_precision': average_precision_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'training_time_seconds': training_time
    }
    
    print(f"\nTest Set Performance:")
    print(f"  ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Precision: {metrics['test_precision']:.4f}")
    print(f"  Recall: {metrics['test_recall']:.4f}")
    print(f"  F1 Score: {metrics['test_f1']:.4f}")
    print(f"  Average Precision: {metrics['test_average_precision']:.4f}")
    print(f"  Training Time: {training_time:.1f} seconds")
    
    return best_model, metrics


# %%
def train_all_models_for_target(target_name, X_train, y_train, X_test, y_test, param_grids):
    """Train all models for a single target variable."""
    print(f"\n{'#'*70}")
    print(f"# TARGET: {target_name}")
    print(f"# Train samples: {len(y_train)}, positive rate: {y_train.mean():.3f}")
    print(f"# Test samples: {len(y_test)}, positive rate: {y_test.mean():.3f}")
    print(f"{'#'*70}")
    
    models = {}
    results = []
    
    for model_name, config in param_grids.items():
        try:
            model, metrics = train_and_evaluate(
                X_train, y_train,
                X_test, y_test,
                model_name,
                config['model'],
                config['params'],
                cv_folds=CV_FOLDS
            )
            models[model_name] = model
            metrics['target'] = target_name
            results.append(metrics)
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    return models, results


# %%
# Create output directories
output_dir = Path("../../result/models_ablation_no_scores")
predictions_dir = Path("../../result/predictions_ablation_no_scores")
output_dir.mkdir(parents=True, exist_ok=True)
predictions_dir.mkdir(parents=True, exist_ok=True)

# %%
# Train models for each target
all_models = {}  # {target: {model_name: model}}
all_results = []

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

for target in TARGET_COLS:
    print(f"\n\n{'*'*80}")
    print(f"Processing target: {target}")
    print(f"{'*'*80}")
    
    y_train = train_df[target].values
    y_test = test_df[target].values
    
    models, results = train_all_models_for_target(
        target, X_train, y_train, X_test, y_test, param_grids
    )
    
    for r in results:
        r['n_train'] = len(train_df)
        r['n_test'] = len(test_df)
        r['ablation'] = 'no_medical_scores'
    
    all_models[target] = models
    all_results.extend(results)

# %%
# Create summary results DataFrame
results_df = pd.DataFrame(all_results)

cols_order = [
    'target', 'model_name', 'ablation', 'cv_roc_auc', 'test_roc_auc', 
    'test_accuracy', 'test_precision', 'test_recall', 'test_f1',
    'test_average_precision', 'training_time_seconds', 'n_train', 'n_test', 'best_params'
]
results_df = results_df[[c for c in cols_order if c in results_df.columns]]

print("\n\n" + "="*80)
print("SUMMARY OF ALL RESULTS (ABLATION: NO MEDICAL SCORES)")
print("="*80)
print(results_df.to_string(index=False))

# %%
# Save results
results_df.to_csv("../../result/model_performance_summary_ablation_no_scores.csv", index=False)
print(f"\nResults saved to result/model_performance_summary_ablation_no_scores.csv")

# %%
# Save models
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

# Save feature columns
with open(output_dir / "feature_columns.pkl", 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"Saved: feature_columns.pkl")

# Save excluded columns info
with open(output_dir / "excluded_columns.pkl", 'wb') as f:
    pickle.dump(MEDICAL_SCORE_COLS, f)
print(f"Saved: excluded_columns.pkl")

# Save SHAP background data
X_train_sample = train_df[feature_cols].sample(n=min(100, len(train_df)), random_state=RANDOM_STATE)
X_train_sample.to_parquet(output_dir / "shap_background.parquet", index=False)
print(f"Saved: shap_background.parquet")

for target, models in all_models.items():
    for model_name, model in models.items():
        model_path = output_dir / f"{target}_{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved: {model_path.name}")

# Save label encoders
with open(output_dir / "label_encoders.pkl", 'wb') as f:
    pickle.dump(label_encoders, f)

# Save metadata
model_metadata = {
    'targets': TARGET_COLS,
    'models': list(param_grids.keys()),
    'feature_cols': feature_cols,
    'excluded_cols': MEDICAL_SCORE_COLS,
    'categorical_cols': CATEGORICAL_COLS,
    'id_cols': ID_COLS,
    'random_state': RANDOM_STATE,
    'ablation_type': 'no_medical_scores'
}
with open(output_dir / "model_metadata.pkl", 'wb') as f:
    pickle.dump(model_metadata, f)

# %%
# Save best models
print("\n" + "="*80)
print("SAVING BEST MODELS")
print("="*80)

best_models_dir = output_dir / "best"
best_models_dir.mkdir(parents=True, exist_ok=True)

best_model_info = {}

for target in TARGET_COLS:
    target_results = results_df[results_df['target'] == target]
    
    if len(target_results) == 0:
        continue
        
    best_idx = target_results['test_roc_auc'].idxmax()
    best_row = target_results.loc[best_idx]
    best_model_name = best_row['model_name']
    best_roc_auc = best_row['test_roc_auc']
    
    best_model = all_models[target][best_model_name]
    
    best_model_path = best_models_dir / f"{target}_best.pkl"
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    best_model_info[target] = {
        'model_name': best_model_name,
        'test_roc_auc': best_roc_auc,
        'cv_roc_auc': best_row['cv_roc_auc'],
        'best_params': best_row['best_params'],
        'model_path': str(best_model_path)
    }
    
    print(f"  {target}: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")

with open(best_models_dir / "best_model_info.pkl", 'wb') as f:
    pickle.dump(best_model_info, f)

# %%
# Create comparison with full model (with medical scores)
print("\n" + "="*80)
print("COMPARISON: Full Model vs Ablation (No Medical Scores)")
print("="*80)

# Load full model results
full_results_path = Path("../../result/model_performance_summary.csv")
if full_results_path.exists():
    full_results_df = pd.read_csv(full_results_path)
    
    comparison_data = []
    for _, row in results_df.iterrows():
        full_match = full_results_df[
            (full_results_df['target'] == row['target']) &
            (full_results_df['model_name'] == row['model_name'])
        ]
        
        if len(full_match) > 0:
            comparison_data.append({
                'Target': row['target'],
                'Model': row['model_name'],
                'Full Model AUROC': full_match.iloc[0]['test_roc_auc'],
                'Ablation AUROC': row['test_roc_auc'],
                'Difference': row['test_roc_auc'] - full_match.iloc[0]['test_roc_auc'],
                'Percent Change': ((row['test_roc_auc'] - full_match.iloc[0]['test_roc_auc']) / full_match.iloc[0]['test_roc_auc']) * 100
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("../../result/model_comparison_ablation_no_scores.csv", index=False)
    print("\nComparison saved to result/model_comparison_ablation_no_scores.csv")
    
    # Display summary
    print("\nAUROC Difference (Ablation - Full Model):")
    print("Negative values indicate performance drop when removing medical scores")
    pivot = comparison_df.pivot_table(
        index='Model', 
        columns='Target', 
        values='Difference', 
        aggfunc='mean'
    )
    print(pivot.round(4).to_string())
    
    print("\n\nPercent Change in AUROC:")
    pivot_pct = comparison_df.pivot_table(
        index='Model', 
        columns='Target', 
        values='Percent Change', 
        aggfunc='mean'
    )
    print(pivot_pct.round(2).to_string())

# %%
print("\n\n" + "="*80)
print("TRAINING COMPLETE - ABLATION STUDY (NO MEDICAL SCORES)")
print("="*80)
print(f"\nExcluded features: {MEDICAL_SCORE_COLS}")
print(f"\nOutput files:")
print(f"  - Results: result/model_performance_summary_ablation_no_scores.csv")
print(f"  - Models: result/models_ablation_no_scores/{{target}}_{{model}}.pkl")
print(f"  - Best models: result/models_ablation_no_scores/best/{{target}}_best.pkl")
print(f"  - Feature columns: result/models_ablation_no_scores/feature_columns.pkl")
print(f"  - SHAP background: result/models_ablation_no_scores/shap_background.parquet")
print(f"  - Comparison: result/model_comparison_ablation_no_scores.csv")
print(f"\nTargets processed: {TARGET_COLS}")
print(f"Models trained: {list(param_grids.keys())}")
