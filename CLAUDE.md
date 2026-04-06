# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Clinical ML project predicting ICU patient outcomes (mortality, infection markers) using MIMIC-IV data with transfusion features. Two cohorts are analyzed: full cohort (all patients) and transfused-only cohort.

## Common Commands

### Full Pipeline (in order)
```bash
# 1. Preprocess raw data → train/test split
cd code/preprocess && python preprocess_input.py

# 2. Train models (both cohorts)
cd code/train && bash train.sh

# 3. Ablation study (both cohorts)
cd code/ablation_study && bash run_ablation_study.sh
```

### Individual Scripts
```bash
# Train full cohort only
python code/train/sklearn_train.py

# Train transfused cohort only
python code/train/sklearn_train_transfused.py

# Ablation: full cohort only
cd code/ablation_study && bash run_ablation_full_cohort.sh

# Ablation: transfused cohort only
cd code/ablation_study && bash run_ablation_transfused_cohort.sh

# Visualizations (run from code/visualization/)
python plot_roc_prc_curves.py
python plot_shap_beeswarm.py

# Analysis tables (run from code/analysis/)
python table_one.py
python model_performance_table.py
```

### No test suite exists. Validation is done via 5-fold stratified cross-validation during training.

## Architecture

**Pipeline flow:** `data/raw/` → `preprocess/` → `data/processed/` → `train/` → `result/`

- **Preprocessing** (`preprocess_input.py`): 80/20 stratified split, median imputation for numerics (with `_missing` indicators), zero imputation for transfusion columns. Split happens before imputation to prevent leakage.
- **Training** (`sklearn_train.py` / `sklearn_train_transfused.py`): GridSearchCV with 5-fold StratifiedKFold over 5 algorithms (DecisionTree, RandomForest, HistGradientBoosting, CatBoost, ANN/MLP). Scoring metric: ROC-AUC.
- **Ablation** (`ablation_study/`): Same pipeline but excludes SOFA components and APACHE2 score to measure their contribution.
- **Visualization** (`visualization/`): ROC/PRC curves, SHAP beeswarm plots, CONSORT flowchart.
- **Analysis** (`analysis/`): Table One (demographics), model performance tables, transfusion pattern tables.

**Cohort pattern:** Most scripts have a `_transfused` variant that filters to patients with any transfusion volume > 0. Results go to `*_transfused/` directories.

## Key Constants

- `RANDOM_STATE = 1004` — used everywhere for reproducibility
- 5 target outcomes: `icu_mortality`, `hospital_mortality`, `mortality_28day`, `blood_culture_positive_after_72h`, `new_antibiotic_after_72h`
- 5 model types: DecisionTree, RandomForest, HistGradientBoosting, CatBoost, ANN (MLP)
- All scripts use relative paths from their own directory (e.g., `../../data/processed/`)

## Dependencies

pandas, numpy, scikit-learn, catboost, shap, matplotlib, seaborn, tableone, pyarrow

## Data

`data/` and `result/` are gitignored. Raw data is `data/raw/analysis_data.parquet` (MIMIC-IV sourced). Processed outputs go to `data/processed/`. All model artifacts, figures, and tables go to `result/`.
