# %%
"""
Classification Model Training with GridSearchCV
================================================
Algorithms: Decision Tree, Random Forest, HistGradientBoosting, CatBoost, ANN (MLP)
Hyperparameter tuning with 5-fold cross-validation
Predictions saved for all targets on the entire dataset

Uses all transfusion timeframes (24h, 48h, 72h) as features in a single model.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
N_JOBS = -1  # Use all available cores

# Define target columns
TARGET_COLS = [
    'icu_mortality',
    'hospital_mortality', 
    'mortality_28day',
    'blood_culture_positive_after_72h',
    'new_antibiotic_after_72h'
]

# ID columns (to be excluded from features)
ID_COLS = ['subject_id', 'hadm_id']

# Categorical columns
CATEGORICAL_COLS = ['surgery_type']

# Transfusion column patterns (all timeframes used)
TRANSFUSION_TYPES = ['rbc_volume', 'ffp_volume', 'platelet_volume']
TIMEFRAMES = ['24h', '48h', '72h']

# %%
# Load data
print("Loading data...")
train_df_orig = pd.read_parquet("../../data/processed/train_data.parquet")
test_df_orig = pd.read_parquet("../../data/processed/test_data.parquet")

print(f"Train shape: {train_df_orig.shape}")
print(f"Test shape: {test_df_orig.shape}")

# %%
# Encode categorical columns (do this once on the original data)
label_encoders = {}
for col in CATEGORICAL_COLS:
    if col in train_df_orig.columns:
        le = LabelEncoder()
        train_df_orig[col] = le.fit_transform(train_df_orig[col].astype(str))
        test_df_orig[col] = le.transform(test_df_orig[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {le.classes_}")


# %%
def get_feature_cols(df):
    """
    Get feature columns using all transfusion timeframes (24h, 48h, 72h).
    """
    exclude_cols = ID_COLS + TARGET_COLS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


# %%
# Show feature columns
feature_cols = get_feature_cols(train_df_orig)
transfusion_cols = [c for c in feature_cols if any(t in c for t in TRANSFUSION_TYPES)]
print(f"\nTotal features: {len(feature_cols)}")
print(f"Transfusion features: {transfusion_cols}")

# %%
# Define hyperparameter grids for each algorithm
# Note: Reduced grid sizes for faster training while keeping impactful parameters
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
    """
    Train model with GridSearchCV and evaluate on test set.
    
    Returns:
        best_model: Trained model with best parameters
        results: Dictionary containing evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=cv,
        scoring='roc_auc',
        n_jobs=N_JOBS,
        verbose=1,
        refit=True
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV ROC-AUC: {best_cv_score:.4f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
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
    """
    Train all models for a single target variable.
    
    Returns:
        models: Dictionary of trained models
        results: List of result dictionaries
    """
    print(f"\n{'#'*70}")
    print(f"# TARGET: {target_name}")
    print(f"# Train positive rate: {y_train.mean():.3f}")
    print(f"# Test positive rate: {y_test.mean():.3f}")
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
output_dir = Path("../../result/models")
predictions_dir = Path("../../result/predictions")
output_dir.mkdir(parents=True, exist_ok=True)
predictions_dir.mkdir(parents=True, exist_ok=True)

# %%
# Train models for each target (using all timeframes as features)
all_models = {}  # {target: {model_name: model}}
all_results = []

# Get feature columns (all timeframes included)
feature_cols = get_feature_cols(train_df_orig)
print(f"\nNumber of features: {len(feature_cols)}")

# Prepare features
X_train = train_df_orig[feature_cols].values
X_test = test_df_orig[feature_cols].values

for target in TARGET_COLS:
    print(f"\n\n{'*'*80}")
    print(f"Processing target: {target}")
    print(f"{'*'*80}")
    
    y_train = train_df_orig[target].values
    y_test = test_df_orig[target].values
    
    models, results = train_all_models_for_target(
        target, X_train, y_train, X_test, y_test, param_grids
    )
    
    all_models[target] = models
    all_results.extend(results)

# %%
# Create summary results DataFrame
results_df = pd.DataFrame(all_results)

# Reorder columns for better readability
cols_order = [
    'target', 'model_name', 'cv_roc_auc', 'test_roc_auc', 
    'test_accuracy', 'test_precision', 'test_recall', 'test_f1',
    'test_average_precision', 'training_time_seconds', 'best_params'
]
results_df = results_df[[c for c in cols_order if c in results_df.columns]]

# Display results
print("\n\n" + "="*80)
print("SUMMARY OF ALL RESULTS")
print("="*80)
print(results_df.to_string(index=False))

# %%
# Save results summary
results_df.to_csv("../../result/model_performance_summary.csv", index=False)
print(f"\nResults saved to result/model_performance_summary.csv")

# %%
# Create pivot table for easy comparison
pivot_auroc = results_df.pivot(index='model_name', columns='target', values='test_roc_auc')
print(f"\n\nTest ROC-AUC by Model and Target:")
print(pivot_auroc.round(4).to_string())

pivot_auroc.to_csv("../../result/model_auroc_comparison.csv")

# %%
# Save trained models and metadata for SHAP analysis
print("\n\n" + "="*80)
print("SAVING MODELS FOR SHAP ANALYSIS")
print("="*80)

# Get feature columns (all timeframes)
feature_cols = get_feature_cols(train_df_orig)

# Save feature columns (needed for SHAP)
feature_cols_path = output_dir / "feature_columns.pkl"
with open(feature_cols_path, 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"Saved feature columns: {feature_cols_path}")

# Save a sample of training data for SHAP background (100 samples)
X_train_sample = train_df_orig[feature_cols].sample(n=min(100, len(train_df_orig)), random_state=RANDOM_STATE)
background_path = output_dir / "shap_background.parquet"
X_train_sample.to_parquet(background_path, index=False)
print(f"Saved SHAP background data: {background_path}")

for target, models in all_models.items():
    for model_name, model in models.items():
        model_path = output_dir / f"{target}_{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model: {model_path}")

# Save label encoders
with open(output_dir / "label_encoders.pkl", 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"Saved: {output_dir / 'label_encoders.pkl'}")

# Save model metadata (for easy loading in SHAP analysis)
model_metadata = {
    'targets': TARGET_COLS,
    'models': list(param_grids.keys()),
    'feature_cols': feature_cols,
    'categorical_cols': CATEGORICAL_COLS,
    'id_cols': ID_COLS,
    'random_state': RANDOM_STATE
}
with open(output_dir / "model_metadata.pkl", 'wb') as f:
    pickle.dump(model_metadata, f)
print(f"Saved: {output_dir / 'model_metadata.pkl'}")

# %%
# Save BEST model for each target (for SHAP analysis)
print("\n\n" + "="*80)
print("SAVING BEST MODELS FOR EACH TARGET")
print("="*80)

best_models_dir = output_dir / "best"
best_models_dir.mkdir(parents=True, exist_ok=True)

best_model_info = {}  # Store info about best models

for target in TARGET_COLS:
    # Find best model for this target
    target_results = results_df[results_df['target'] == target]
    best_idx = target_results['test_roc_auc'].idxmax()
    best_row = target_results.loc[best_idx]
    best_model_name = best_row['model_name']
    best_roc_auc = best_row['test_roc_auc']
    
    # Get the best model
    best_model = all_models[target][best_model_name]
    
    # Save best model
    best_model_path = best_models_dir / f"{target}_best.pkl"
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Store best model info
    best_model_info[target] = {
        'model_name': best_model_name,
        'test_roc_auc': best_roc_auc,
        'cv_roc_auc': best_row['cv_roc_auc'],
        'best_params': best_row['best_params'],
        'model_path': str(best_model_path)
    }
    
    print(f"  {target}: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")

# Save best model info summary
with open(best_models_dir / "best_model_info.pkl", 'wb') as f:
    pickle.dump(best_model_info, f)

# Also save as CSV for easy viewing
best_model_summary = []
for target, info in best_model_info.items():
    best_model_summary.append({
        'target': target,
        'best_model': info['model_name'],
        'test_roc_auc': info['test_roc_auc'],
        'cv_roc_auc': info['cv_roc_auc']
    })

best_model_df = pd.DataFrame(best_model_summary)
best_model_df.to_csv(best_models_dir / "best_model_summary.csv", index=False)
print(f"\nSaved best model summary: {best_models_dir / 'best_model_summary.csv'}")

# %%
# Generate predictions on the ENTIRE dataset (train + test combined)
print("\n\n" + "="*80)
print("GENERATING PREDICTIONS ON ENTIRE DATASET")
print("="*80)

# Combine train and test data
full_df = pd.concat([train_df_orig, test_df_orig], axis=0).reset_index(drop=True)
print(f"Full dataset shape: {full_df.shape}")

# Get feature columns (all timeframes)
feature_cols = get_feature_cols(full_df)
X_full = full_df[feature_cols].values

# Create predictions DataFrame with IDs
predictions_df = full_df[ID_COLS].copy()

# Add actual target values
for target in TARGET_COLS:
    predictions_df[f'{target}_actual'] = full_df[target].values

# Add predictions from each model for each target
for target in TARGET_COLS:
    print(f"\nGenerating predictions for {target}...")
    
    for model_name, model in all_models[target].items():
        # Predict probabilities (class 1 probability)
        y_pred_proba = model.predict_proba(X_full)[:, 1]
        y_pred_class = model.predict(X_full)
        
        predictions_df[f'{target}_{model_name}_proba'] = y_pred_proba
        predictions_df[f'{target}_{model_name}_pred'] = y_pred_class
        
        print(f"  - {model_name}: done")

# Save predictions
predictions_path = predictions_dir / "all_predictions.parquet"
predictions_df.to_parquet(predictions_path, index=False)
print(f"\nPredictions saved to: {predictions_path}")

# Also save as CSV for easy viewing
predictions_df.to_csv(predictions_dir / "all_predictions.csv", index=False)
print(f"Predictions also saved to: {predictions_dir / 'all_predictions.csv'}")

# Display prediction summary
print(f"\nPrediction Columns:")
print(predictions_df.columns.tolist())
print(f"\nPrediction DataFrame shape: {predictions_df.shape}")

# %%
# Find best model for each target
print("\n\n" + "="*80)
print("BEST MODEL FOR EACH TARGET (by Test ROC-AUC)")
print("="*80)

for target in TARGET_COLS:
    target_results = results_df[results_df['target'] == target]
    best_idx = target_results['test_roc_auc'].idxmax()
    best_row = target_results.loc[best_idx]
    print(f"\n{target}:")
    print(f"  Best Model: {best_row['model_name']}")
    print(f"  Test ROC-AUC: {best_row['test_roc_auc']:.4f}")
    print(f"  CV ROC-AUC: {best_row['cv_roc_auc']:.4f}")

# %%
print("\n\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nOutput files:")
print(f"  - Model performance summary: result/model_performance_summary.csv")
print(f"  - ROC-AUC comparison: result/model_auroc_comparison.csv")
print(f"  - Trained models: result/models/{{target}}_{{model}}.pkl")
print(f"  - Predictions (parquet): result/predictions/all_predictions.parquet")
print(f"  - Predictions (csv): result/predictions/all_predictions.csv")
print(f"\nFiles for SHAP analysis:")
print(f"  - Best models: result/models/best/{{target}}_best.pkl")
print(f"  - Best model summary: result/models/best/best_model_summary.csv")
print(f"  - Best model info: result/models/best/best_model_info.pkl")
print(f"  - Feature columns: result/models/feature_columns.pkl")
print(f"  - SHAP background data: result/models/shap_background.parquet")
print(f"  - Model metadata: result/models/model_metadata.pkl")
print(f"\nTargets processed: {TARGET_COLS}")
print(f"Models trained: {list(param_grids.keys())}")


# %%
