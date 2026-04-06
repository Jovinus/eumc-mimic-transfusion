# Ablation Study: Model Performance without Medical Scores

## Overview

This ablation study evaluates the contribution of medical severity scores (SOFA and APACHE2) to model prediction performance. By training models without these clinical scoring systems, we can understand their importance in predicting patient outcomes.

The study is conducted on **two cohorts**:
1. **Full Cohort** - All patients (with and without transfusion)
2. **Transfused Cohort** - Only patients who received any transfusion

## Excluded Features

The following features are excluded from model training:

### SOFA Score Components
- `sofa_resp` - Respiratory component
- `sofa_coag` - Coagulation component
- `sofa_liver` - Liver component
- `sofa_cardio` - Cardiovascular component
- `sofa_cns` - Central nervous system component
- `sofa_renal` - Renal component
- `sofa_total` - Total SOFA score

### APACHE2 Score
- `apache2_score` - APACHE II severity score

### Missing Indicators
- All corresponding `*_missing` columns for the above features

## Files

### Full Cohort Scripts
| File | Description |
|------|-------------|
| `sklearn_train_no_scores.py` | Train models on full cohort without medical scores |
| `plot_roc_prc_curves_no_scores.py` | Generate ROC/PRC curves for full cohort |
| `plot_shap_beeswarm_no_scores.py` | Generate SHAP plots for full cohort |

### Transfused Cohort Scripts
| File | Description |
|------|-------------|
| `sklearn_train_no_scores_transfused.py` | Train models on transfused cohort without medical scores |
| `plot_roc_prc_curves_no_scores_transfused.py` | Generate ROC/PRC curves for transfused cohort |
| `plot_shap_beeswarm_no_scores_transfused.py` | Generate SHAP plots for transfused cohort |

### Run Script
- `run_ablation_study.sh` - Runs the complete ablation study pipeline for both cohorts

## Usage

### Run Complete Pipeline (Both Cohorts)
```bash
cd code/ablation_study
bash run_ablation_study.sh
```

### Run Full Cohort Only
```bash
python sklearn_train_no_scores.py
python plot_roc_prc_curves_no_scores.py
python plot_shap_beeswarm_no_scores.py
```

### Run Transfused Cohort Only
```bash
python sklearn_train_no_scores_transfused.py
python plot_roc_prc_curves_no_scores_transfused.py
python plot_shap_beeswarm_no_scores_transfused.py
```

## Output

### Full Cohort Results
| Output | Location |
|--------|----------|
| Performance metrics | `result/model_performance_summary_ablation_no_scores.csv` |
| Comparison with full model | `result/model_comparison_ablation_no_scores.csv` |
| Trained models | `result/models_ablation_no_scores/` |
| Best models | `result/models_ablation_no_scores/best/` |
| ROC/PRC curves | `result/figures_ablation_no_scores/curves/` |
| SHAP plots | `result/figures_ablation_no_scores/shap/` |

### Transfused Cohort Results
| Output | Location |
|--------|----------|
| Performance metrics | `result/model_performance_summary_ablation_no_scores_transfused.csv` |
| Comparison with full model | `result/model_comparison_ablation_no_scores_transfused.csv` |
| Trained models | `result/models_ablation_no_scores_transfused/` |
| Best models | `result/models_ablation_no_scores_transfused/best/` |
| ROC/PRC curves | `result/figures_ablation_no_scores_transfused/curves/` |
| SHAP plots | `result/figures_ablation_no_scores_transfused/shap/` |

## Interpretation

The comparison between full model and ablation model AUROC values indicates the contribution of medical scores:

| Difference | Interpretation |
|------------|----------------|
| **Negative** | Removing medical scores decreases performance → scores are important predictors |
| **Near-zero** | Medical scores have minimal impact on predictions |
| **Positive** | Removing medical scores improves performance → possible overfitting with scores |

### Expected Analysis Questions

1. **Do medical scores contribute significantly to prediction accuracy?**
   - Compare AUROC between full model and ablation model
   
2. **Is the contribution of medical scores different across outcomes?**
   - Some outcomes may rely more heavily on severity scores
   
3. **Is the contribution different between full cohort and transfused cohort?**
   - Transfused patients may have different characteristics

## Target Outcomes

| Target | Description |
|--------|-------------|
| `icu_mortality` | ICU Mortality |
| `hospital_mortality` | Hospital Mortality |
| `mortality_28day` | 28-Day Mortality |
| `blood_culture_positive_after_72h` | Blood Culture Positive (>72h) |
| `new_antibiotic_after_72h` | New Antibiotic (>72h) |

## Models

| Model | Description |
|-------|-------------|
| Decision Tree | Single decision tree classifier |
| Random Forest | Ensemble of decision trees |
| Histogram Gradient Boosting | Gradient boosting with histogram binning |
| CatBoost | Gradient boosting optimized for categorical features |
| ANN (MLP) | Multi-layer perceptron neural network |

## Comparison Summary

After running the ablation study, you can compare results across:

1. **Full Model vs Ablation** - Effect of removing medical scores
2. **Full Cohort vs Transfused Cohort** - How the ablation effect differs by patient population
3. **Across Targets** - Which outcomes are most affected by medical scores
4. **Across Models** - Whether certain model types are more robust to feature removal

The comparison CSV files provide:
- Full Model AUROC
- Ablation AUROC
- Absolute Difference
- Percent Change
