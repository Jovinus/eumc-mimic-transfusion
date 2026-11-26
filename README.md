# Early transfusion volume and clinical outcomes in critically ill patients

Analysis code for the manuscript *"Early transfusion volume and clinical outcomes in critically ill patients: a retrospective cohort study using explainable machine learning"* (MIMIC-IV v3.1).

This repository contains code only. MIMIC-IV data, and any file derived from it, are not included and must not be committed (PhysioNet credentialed data use agreement). Obtain MIMIC-IV v3.1 from <https://physionet.org/content/mimiciv/3.1/>.

## Contents

| Folder | What it does |
|---|---|
| `extraction/` | Converts MIMIC-IV v3.1 to parquet and extracts the admission-level analysis dataset. Variable definitions are in `extraction/docs/variable_definitions.md`; corrections relative to the first extraction are in `extraction/docs/v2_changes.md`. |
| `analysis/` | Builds the final cohort file, trains and evaluates the models, and produces the tables and figures. |

## Requirements

Python 3.12 and the packages in `requirements.txt`. The extraction step also needs `duckdb`.

## Directory layout expected by the scripts

`<BASE>` is any working directory. It is passed as the first argument to every script in `analysis/`.

```
<BASE>/
  extraction/
    data/mimic-iv-parquet/{hosp,icu}/*.parquet        # output of extraction step 1
    reference_output/v2_corrected/analysis_data.parquet   # output of extraction step 3
  out/        # created by the analysis scripts (derived data and result tables)
  outputs/    # created by the analysis scripts (Word tables, figures)
```

## 1. Extraction

```
python extraction/01_convert_to_parquet.py --src <mimic-iv-3.1 dir> --dst <BASE>/extraction/data/mimic-iv-parquet
python extraction/02_feature_selection.py --data-path <BASE>/extraction/data/mimic-iv-parquet --save-path <features dir>
python extraction/03_create_analysis_data.py --data-path <BASE>/extraction/data/mimic-iv-parquet --feature-path <features dir> --save-path <BASE>/extraction/reference_output/v2_corrected
```

Step 1 stores every column as text; the later steps assume this format.

## 2. Analysis

Run in this order. `RUNDIR=v4` selects the final dataset (invasive-ventilation and blood-culture definitions applied in `13_build_v4.py`).

```
python analysis/06_build_v3.py <BASE>          # adds age, sex, CRRT within 24 h, timing flags, patient-level split
python analysis/12_explore_defs.py <BASE>      # extracts ventilation events used by the next step
python analysis/13_build_v4.py <BASE>          # final dataset: out/v4/analysis_v3.parquet

set RUNDIR=v4                                  # Windows; on Linux/macOS: export RUNDIR=v4
python analysis/07_train_v3.py <BASE>          # five models x five outcomes x feature sets (cohorts M, I, S)
python analysis/08_evaluate_v3.py <BASE>       # performance, calibration, ablation ΔAUROC, grouped SHAP
python analysis/09_association_v3.py <BASE>    # logistic regression, cluster-robust SE by patient
python analysis/10_table1_v3.py <BASE>         # baseline characteristics
python analysis/11_make_outputs_v3.py <BASE>   # Word tables and flow diagram
python analysis/14_figures.py <BASE>           # Figures 2-6
```

Cohorts:

- **M** — transfused within 24 h of ICU admission and alive at 24 h (mortality outcomes; 24-h volumes).
- **I** — M and still in hospital at 72 h (positive blood culture and new antibiotic after 72 h; 24-h volumes).
- **S** — transfused within 72 h and still in hospital at 72 h (sensitivity analysis; 24-, 48- and 72-h volumes).

Training and test sets are separated at the patient level (`StratifiedGroupKFold`, seed 1004), and cross-validation is grouped by patient.

## Notes

- All scripts print aggregate results only; none prints patient-level rows.
- Model training (`07_train_v3.py`) is the slow step.
- A large language model was used in writing the code in `analysis/`, as stated in the manuscript. The analysis was independently repeated by one of the authors.
