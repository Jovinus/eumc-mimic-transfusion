"""Re-analysis training on the corrected dataset (v3).
Analyses (cohort, transfusion predictors, outcomes):
  M : transfused <=24h & alive at 24h          | 24h volumes (3) | 3 mortality outcomes
  I : M & still in hospital at 72h             | 24h volumes (3) | blood culture >72h, new antibiotic >72h
  S : transfused <=72h & in hospital at 72h    | 24/48/72h (9)   | all 5 outcomes (sensitivity, 72h landmark)
Feature sets: full, no_transfusion, no_scores (S: full, no_transfusion).
Patient-level split and patient-grouped 5-fold CV. Original hyperparameter grids.
Usage: python 07_train_v3.py <project root> [M I S]
"""
import sys, pickle, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
RS = 1004
MORT = ["icu_mortality", "hospital_mortality", "mortality_28day"]
INF = ["blood_culture_positive_after_72h", "new_antibiotic_after_72h"]
TF24 = ["rbc_volume_24h", "ffp_volume_24h", "platelet_volume_24h"]
TF = [f"{p}_volume_{t}" for p in ("rbc", "ffp", "platelet") for t in ("24h", "48h", "72h")]
BASEF = ["age", "male", "a_sbp_24h_mean", "a_dbp_24h_mean", "pulse_24h_mean", "resp_24h_mean", "spo2_24h_mean",
         "creatinine_first", "wbc_first", "crp_first", "hemoglobin_first", "hematocrit_first", "platelet_first",
         "pt_inr_first", "aptt_first", "urine_output_24h_cumulative", "is_elective_surgery", "is_emergency_surgery",
         "mech_vent_24h", "norepinephrine_24h", "crrt_24h"]
SCORES = ["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns", "sofa_renal", "sofa_total", "apache2_score"]
ANALYSES = {"M": ("cohort_M", TF24, MORT, ["full", "no_transfusion", "no_scores"]),
            "I": ("cohort_I", TF24, INF, ["full", "no_transfusion", "no_scores"]),
            "S": ("cohort_S", TF, MORT + INF, ["full", "no_transfusion"])}


def grids():
    return {
        "DecisionTree": (DecisionTreeClassifier(random_state=RS),
                         {"max_depth": [3, 5, 7, 10], "min_samples_split": [2, 5, 10],
                          "min_samples_leaf": [1, 2, 5], "criterion": ["gini", "entropy"]}),
        "RandomForest": (RandomForestClassifier(random_state=RS, n_jobs=1),
                         {"n_estimators": [100, 200], "max_depth": [5, 10, 15], "min_samples_split": [2, 5],
                          "min_samples_leaf": [1, 2], "max_features": ["sqrt", "log2"]}),
        "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=RS),
                                 {"learning_rate": [0.05, 0.1], "max_iter": [100, 200], "max_depth": [5, 7, 10],
                                  "min_samples_leaf": [20, 50], "l2_regularization": [0, 0.1]}),
        "CatBoost": (CatBoostClassifier(random_state=RS, verbose=0, thread_count=1, allow_writing_files=False),
                     {"learning_rate": [0.05, 0.1], "depth": [4, 6, 8], "iterations": [100, 200], "l2_leaf_reg": [1, 3, 5]}),
        "ANN": (Pipeline([("scaler", StandardScaler()),
                          ("mlp", MLPClassifier(random_state=RS, max_iter=300, early_stopping=True))]),
                {"mlp__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)], "mlp__activation": ["relu"],
                 "mlp__alpha": [0.001, 0.01], "mlp__learning_rate_init": [0.001, 0.01], "mlp__batch_size": [64, 128]}),
    }


def prepare(d, cohort, tfv):
    """median imputation (training medians) + missing indicators; transfusion NaN -> 0."""
    s = d[d[cohort] == 1].reset_index(drop=True)
    tr = s.split == "train"
    X = s[BASEF + SCORES + tfv].astype(float).copy()
    X[tfv] = X[tfv].fillna(0)
    for c in BASEF + SCORES:
        if X.loc[tr, c].isna().any():
            X[c + "_missing"] = X[c].isna().astype(int)
            X[c] = X[c].fillna(X.loc[tr, c].median())
        elif X[c].isna().any():
            X[c] = X[c].fillna(X.loc[tr, c].median())
    X = X.loc[:, X.loc[tr].nunique() > 1]
    return s, X, tr.values


def main(base, which):
    OUT = Path(base) / "out" / __import__("os").environ.get("RUNDIR", "v3")
    d = pd.read_parquet(OUT / "analysis_v3.parquet")
    for a in which:
        cohort, tfv, targets, fsets = ANALYSES[a]
        s, X, tr = prepare(d, cohort, tfv)
        X.assign(split=np.where(tr, "train", "test"), hadm_id=s.hadm_id).to_parquet(OUT / f"X_{a}.parquet", index=False)
        groups = s.subject_id.values[tr]
        cols = {"full": list(X.columns),
                "no_transfusion": [c for c in X.columns if c not in tfv],
                "no_scores": [c for c in X.columns if c.replace("_missing", "") not in SCORES]}
        for fs in fsets:
            feat = cols[fs]
            mdir = OUT / f"models_{a}_{fs}"; mdir.mkdir(exist_ok=True)
            with open(mdir / "feature_columns.pkl", "wb") as f:
                pickle.dump(feat, f)
            Xtr, Xte = X.loc[tr, feat].values, X.loc[~tr, feat].values
            test_pred = s.loc[~tr, ["subject_id", "hadm_id"] + targets].reset_index(drop=True)
            oof_pred = s.loc[tr, ["subject_id", "hadm_id"] + targets].reset_index(drop=True)
            rows = []
            print(f"##### {a}/{fs}: {len(feat)} features, train {Xtr.shape[0]}, test {Xte.shape[0]}", flush=True)
            for t in targets:
                ytr, yte = s.loc[tr, t].values, s.loc[~tr, t].values
                cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS).split(Xtr, ytr, groups))
                for name, (est, grid) in grids().items():
                    t0 = time.time()
                    gs = GridSearchCV(est, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
                    gs.fit(Xtr, ytr)
                    p = gs.best_estimator_.predict_proba(Xte)[:, 1]
                    oof = cross_val_predict(clone(gs.best_estimator_), Xtr, ytr, cv=cv, method="predict_proba", n_jobs=5)[:, 1]
                    test_pred[f"{t}__{name}"] = p
                    oof_pred[f"{t}__{name}"] = oof
                    with open(mdir / f"{t}__{name}.pkl", "wb") as f:
                        pickle.dump(gs.best_estimator_, f)
                    rows.append({"analysis": a, "feature_set": fs, "target": t, "model": name, "cv_auc": gs.best_score_,
                                 "test_auc": roc_auc_score(yte, p), "oof_auc": roc_auc_score(ytr, oof),
                                 "best_params": str(gs.best_params_), "sec": round(time.time() - t0, 1)})
                    print(f"{a}/{fs:15s} {t:34s} {name:22s} cv {gs.best_score_:.4f} test {rows[-1]['test_auc']:.4f} ({rows[-1]['sec']}s)", flush=True)
                    test_pred.to_parquet(OUT / f"pred_test_{a}_{fs}.parquet", index=False)
                    oof_pred.to_parquet(OUT / f"pred_oof_{a}_{fs}.parquet", index=False)
                    pd.DataFrame(rows).to_csv(OUT / f"train_log_{a}_{fs}.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["M", "I", "S"])
