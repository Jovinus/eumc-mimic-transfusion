"""Evaluate the v3 re-analysis: performance (bootstrap CI, train-derived threshold, calibration),
ablation ΔAUROC (paired bootstrap), grouped SHAP. Aggregates only."""
import sys, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import statsmodels.api as sm
import shap

warnings.filterwarnings("ignore")
OUT = Path(sys.argv[1]) / "out" / __import__("os").environ.get("RUNDIR", "v3")
MORT = ["icu_mortality", "hospital_mortality", "mortality_28day"]
INF = ["blood_culture_positive_after_72h", "new_antibiotic_after_72h"]
AN = {"M": (MORT, ["no_transfusion", "no_scores"]), "I": (INF, ["no_transfusion", "no_scores"]), "S": (MORT + INF, ["no_transfusion"])}
MODELS = ["DecisionTree", "RandomForest", "HistGradientBoosting", "CatBoost", "ANN"]
SCORES = ["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns", "sofa_renal", "sofa_total", "apache2_score"]
B = 1000
rng = np.random.default_rng(1004)


def boot_idx(y):
    out = []
    while len(out) < B:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(i)
    return out


def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    r = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    return r.params[1], r.params[0]


perf, abl, shp = [], [], []
for a, (targets, ablations) in AN.items():
    te = pd.read_parquet(OUT / f"pred_test_{a}_full.parquet"); oof = pd.read_parquet(OUT / f"pred_oof_{a}_full.parquet")
    alt = {fs: pd.read_parquet(OUT / f"pred_test_{a}_{fs}.parquet") for fs in ablations}
    for fs in ablations:
        assert (alt[fs].hadm_id.values == te.hadm_id.values).all()
    for t in targets:
        y = te[t].values.astype(int); ytr = oof[t].values.astype(int)
        idx = boot_idx(y)
        for m in MODELS:
            p = te[f"{t}__{m}"].values; po = oof[f"{t}__{m}"].values
            auc = roc_auc_score(y, p)
            bs = np.array([roc_auc_score(y[i], p[i]) for i in idx])
            fpr, tpr, thr = roc_curve(ytr, po)
            th = thr[np.argmax(tpr - fpr)]
            pred = p >= th
            tp, fp, fn, tn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
            slope, inter = calib(y, p)
            perf.append({"Analysis": a, "Outcome": t, "Model": m, "N_test": len(y), "Events": int(y.sum()),
                         "AUROC": auc, "AUROC_lo": np.percentile(bs, 2.5), "AUROC_hi": np.percentile(bs, 97.5),
                         "AUPRC": average_precision_score(y, p), "Brier": brier_score_loss(y, p),
                         "Calibration slope": slope, "Calibration intercept": inter, "Threshold (train OOF Youden)": th,
                         "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp), "PPV": tp / max(tp + fp, 1), "NPV": tn / max(tn + fn, 1)})
            for fs in ablations:
                q = alt[fs][f"{t}__{m}"].values
                dlt = np.array([roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], q[i]) for i in idx])
                abl.append({"Analysis": a, "Outcome": t, "Model": m, "ablation": fs, "AUROC_full": auc, "AUROC_ablated": roc_auc_score(y, q),
                            "delta": auc - roc_auc_score(y, q), "ci_lo": np.percentile(dlt, 2.5), "ci_hi": np.percentile(dlt, 97.5)})
    # grouped SHAP (tree models, full feature set, whole test set, log-odds / raw scale)
    X = pd.read_parquet(OUT / f"X_{a}.parquet")
    feat = pickle.load(open(OUT / f"models_{a}_full" / "feature_columns.pkl", "rb"))
    Xte = X.loc[X.split == "test", feat]

    def group(c):
        b = c.replace("_missing", "")
        if "_volume_" in b: return "Transfusion volumes"
        if b in SCORES: return "SOFA / APACHE II"
        if b in ("age", "male"): return "Age / sex"
        if b.endswith("_first"): return "Laboratory"
        if b.endswith("_24h_mean") or b.startswith("urine"): return "Vital signs / urine output"
        if b.startswith("is_"): return "Surgery type"
        return "Organ support (MV, norepinephrine, CRRT)"
    grp = np.array([group(c) for c in feat])
    for t in targets:
        for m in ["RandomForest", "HistGradientBoosting", "CatBoost"]:
            mdl = pickle.load(open(OUT / f"models_{a}_full" / f"{t}__{m}.pkl", "rb"))
            sv = shap.TreeExplainer(mdl).shap_values(Xte.values, check_additivity=False)
            if isinstance(sv, list): sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3: sv = sv[:, :, 1]
            tot = {g: np.abs(sv[:, grp == g].sum(axis=1)).mean() for g in np.unique(grp)}
            ssum = sum(tot.values())
            single = pd.Series(np.abs(sv).mean(axis=0), index=feat).sort_values(ascending=False)
            for g, v in tot.items():
                shp.append({"Analysis": a, "Outcome": t, "Model": m, "Group": g, "mean_abs_grouped_shap": v, "share_pct": 100 * v / ssum,
                            "rank": 1 + sum(x > v for x in tot.values()), "top5_single_features": "; ".join(single.index[:5]),
                            "best_rank_of_transfusion_feature": 1 + min(list(single.index).index(c) for c in feat if "_volume_" in c)})
        print("SHAP done", a, t, flush=True)

perf = pd.DataFrame(perf); abl = pd.DataFrame(abl); shp = pd.DataFrame(shp)
perf.to_csv(OUT / "B_performance.csv", index=False); abl.to_csv(OUT / "C_ablation.csv", index=False); shp.to_csv(OUT / "E_grouped_shap.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print(perf[["Analysis", "Outcome", "Model", "Events", "AUROC", "AUROC_lo", "AUROC_hi", "AUPRC", "Brier", "Calibration slope", "Sensitivity", "Specificity"]].to_string(index=False))
print(abl.to_string(index=False))
tf = shp[shp.Group.isin(["Transfusion volumes", "SOFA / APACHE II", "Age / sex"])]
print(tf.pivot_table(index=["Analysis", "Outcome", "Model"], columns="Group", values="share_pct").to_string())
print(shp[shp.Group == "Transfusion volumes"][["Analysis", "Outcome", "Model", "share_pct", "rank", "best_rank_of_transfusion_feature", "top5_single_features"]].to_string(index=False))
