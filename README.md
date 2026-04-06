# Predicting Clinical Outcomes in ICU Patients Using Transfusion Data: A Machine Learning Approach with MIMIC-IV

## Background

Blood transfusion is one of the most common interventions in the intensive care unit (ICU), yet its association with clinical outcomes remains complex and incompletely understood. While transfusion can be life-saving in acute hemorrhage and severe anemia, liberal transfusion strategies have been linked to increased morbidity, including infection, organ dysfunction, and mortality. Understanding how transfusion patterns relate to patient outcomes is essential for optimizing transfusion practice in critically ill patients.

This project develops machine learning models to predict five clinically important outcomes in ICU patients using data from the MIMIC-IV database, with a focus on the role of blood product transfusion volumes across multiple time windows.

## Study Design

### Data Source

- **Database:** MIMIC-IV (Medical Information Mart for Intensive Care, version IV)
- **Population:** Adult ICU patients
- **Cohorts:**
  - **Full Cohort** — All eligible ICU patients (with and without transfusion)
  - **Transfused Cohort** — Subset of patients who received any blood product transfusion (RBC, FFP, or platelet)

### Target Outcomes

| Outcome | Description |
|---------|-------------|
| ICU Mortality | Death during ICU stay |
| Hospital Mortality | Death during hospital admission |
| 28-Day Mortality | Death within 28 days of ICU admission |
| Blood Culture Positive (>72h) | New positive blood culture after 72 hours |
| New Antibiotic (>72h) | New antibiotic initiation after 72 hours |

The first three outcomes capture short- and medium-term mortality, while the latter two serve as surrogate markers for ICU-acquired infection, which is a known complication associated with transfusion-related immunomodulation (TRIM).

### Feature Categories

| Category | Features |
|----------|----------|
| Vital Signs | Arterial SBP, DBP, pulse rate, respiratory rate, SpO2 (24h means) |
| Laboratory Values | Creatinine, WBC, CRP, hemoglobin, hematocrit, platelet count, PT-INR, aPTT, urine output |
| Severity Scores | SOFA components (respiratory, coagulation, liver, cardiovascular, CNS, renal, total), APACHE II |
| Surgery Type | Elective surgery, emergency surgery, non-surgical |
| Clinical Interventions | Mechanical ventilation (24h), norepinephrine use (24h), CRRT initiation |
| Transfusion Volumes | RBC, FFP, and platelet volumes at 24h, 48h, and 72h windows |
| Missing Indicators | Binary flags for missingness in numeric features |

### Machine Learning Models

Five classification algorithms are compared:

| Model | Description |
|-------|-------------|
| Decision Tree | Single tree classifier (baseline) |
| Random Forest | Ensemble of decision trees with bagging |
| Histogram Gradient Boosting | Gradient boosting with histogram-based binning |
| CatBoost | Gradient boosting optimized for categorical features |
| ANN (MLP) | Multi-layer perceptron neural network with StandardScaler preprocessing |

All models are tuned via **GridSearchCV** with **5-fold stratified cross-validation**, optimizing for **ROC-AUC**.

### Ablation Study

An ablation study is conducted to assess the contribution of medical severity scores (SOFA and APACHE II) to model performance. Models are retrained after removing all SOFA components, SOFA total, APACHE II score, and their corresponding missing indicators. The difference in AUROC between the full model and the ablated model quantifies the contribution of these clinical scoring systems.

## Project Structure

```
eumc-mimic-transfusion/
├── data/
│   ├── raw/                        # Raw MIMIC-IV data (analysis_data.parquet)
│   └── processed/                  # Preprocessed train/test splits
├── code/
│   ├── preprocess/
│   │   └── preprocess_input.py     # Data preprocessing and train/test split
│   ├── train/
│   │   ├── sklearn_train.py        # Model training (full cohort)
│   │   ├── sklearn_train_transfused.py  # Model training (transfused cohort)
│   │   └── train.sh                # Run both training scripts
│   ├── ablation_study/
│   │   ├── sklearn_train_no_scores*.py  # Training without severity scores
│   │   ├── plot_roc_prc_curves_no_scores*.py
│   │   ├── plot_shap_beeswarm_no_scores*.py
│   │   └── run_ablation_study.sh   # Run complete ablation pipeline
│   ├── analysis/
│   │   ├── table_one*.py           # Baseline characteristics tables
│   │   ├── model_performance_table*.py  # Classification metrics tables
│   │   └── table_transfusion_by_outcome*.py  # Transfusion pattern analysis
│   └── visualization/
│       ├── data_flow_chart.py      # CONSORT-style patient flow diagram
│       ├── plot_roc_prc_curves*.py # ROC and precision-recall curves
│       └── plot_shap_beeswarm*.py  # SHAP feature importance plots
├── result/                         # Output directory (models, figures, tables)
├── CLAUDE.md
└── README.md
```

> **Note:** Files with `*_transfused` suffixes are variants that operate on the transfused-only cohort.

## Methods

### Preprocessing

1. **Train/Test Split:** 80/20 stratified split using a composite stratification variable combining all five target outcomes. Rare outcome combinations (<20 samples) are merged to ensure valid stratification. The split is performed **before** imputation to prevent data leakage.
2. **Missing Value Handling:**
   - **Numeric features:** Median imputation using training set statistics, with binary missing indicator columns (`{feature}_missing`) appended.
   - **Transfusion features:** Imputed with 0 (missing indicates no transfusion was given; no missing indicator needed).
3. **Categorical Encoding:** `surgery_type` is label-encoded.

### Model Training and Evaluation

- Hyperparameter search is performed via `GridSearchCV` with 5-fold `StratifiedKFold` and ROC-AUC scoring.
- The best model per target is selected by test set ROC-AUC and saved separately for downstream analysis.
- Predictions (probabilities and class labels) are generated on the full dataset (train + test combined) for all model-target combinations.

### Model Interpretability

- **SHAP (SHapley Additive exPlanations):** Beeswarm plots visualize feature contributions for each model and target. A 100-sample background dataset from the training set is used for SHAP value computation.
- **ROC/PRC Curves:** All models are plotted on a single figure per outcome for direct comparison.

### Statistical Analysis

- **Table One:** Baseline characteristics compared between train and test sets using pre-imputation values to reflect the original data distribution. Continuous variables are reported as Mean ± SD; categorical variables as n (%).
- **Performance Tables:** Optimal classification thresholds are determined via Youden's J statistic. Metrics include accuracy, precision, recall, specificity, NPV, AUROC, and AUPRC.
- **Transfusion Analysis:** Transfusion volumes (RBC, FFP, platelet) are compared across outcome groups.

## How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn catboost shap matplotlib seaborn tableone pyarrow
```

### Full Pipeline

```bash
# Step 1: Preprocess data
cd code/preprocess
python preprocess_input.py

# Step 2: Train models (both cohorts)
cd ../train
bash train.sh

# Step 3: Generate visualizations
cd ../visualization
python data_flow_chart.py
python plot_roc_prc_curves.py
python plot_roc_prc_curves_transfused.py
python plot_shap_beeswarm.py
python plot_shap_beeswarm_transfused.py

# Step 4: Generate analysis tables
cd ../analysis
python table_one.py
python table_one_transfused.py
python model_performance_table.py
python model_performance_table_transfused.py
python table_transfusion_by_outcome.py
python table_transfusion_by_outcome_transfused.py

# Step 5: Run ablation study
cd ../ablation_study
bash run_ablation_study.sh
```

### Output

| Output Type | Location |
|-------------|----------|
| Trained models | `result/models/`, `result/models_transfused/` |
| Best models (per target) | `result/models/best/`, `result/models_transfused/best/` |
| Performance summary | `result/model_performance_summary.csv` |
| AUROC comparison | `result/model_auroc_comparison.csv` |
| Predictions | `result/predictions/` |
| ROC/PRC curves | `result/figures/curves/`, `result/figures_transfused/curves/` |
| SHAP plots | `result/figures/shap/`, `result/figures_transfused/shap/` |
| Table One | `result/tables/`, `result/tables_transfused/` |
| Ablation comparison | `result/model_comparison_ablation_no_scores*.csv` |

## Reproducibility

- **Random seed:** 1004 (consistent across all scripts)
- **Cross-validation:** 5-fold stratified
- **Model artifacts:** All trained models, feature columns, label encoders, and metadata are serialized as pickle files
- **Imputation statistics:** Saved to `data/processed/imputation_stats.csv` for reproducible inference on new data

## Ethical Considerations

This study uses the MIMIC-IV database, which contains de-identified health data from Beth Israel Deaconess Medical Center. Access to MIMIC-IV requires completion of a recognized human research ethics training course and a signed data use agreement via PhysioNet. All data handling complies with the MIMIC-IV data use agreement.
