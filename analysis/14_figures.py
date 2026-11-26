"""Figures 2-6 for the re-analysis (RUNDIR, default v4)."""
import sys, os, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import shap

warnings.filterwarnings("ignore")
BASE = Path(sys.argv[1]); OUT = BASE / "out" / os.environ.get("RUNDIR", "v4"); REV = BASE / "outputs"
ON = {"icu_mortality": "ICU mortality", "hospital_mortality": "In-hospital mortality", "mortality_28day": "28-day mortality",
      "blood_culture_positive_after_72h": "Positive blood culture >72 h", "new_antibiotic_after_72h": "New antibiotic >72 h"}
MN = {"DecisionTree": "Decision tree", "RandomForest": "Random forest", "HistGradientBoosting": "Hist. gradient boosting", "CatBoost": "CatBoost", "ANN": "MLP"}
COL = dict(zip(MN, ["#7f7f7f", "#2ca02c", "#1f77b4", "#d62728", "#9467bd"]))
PANELS = [("M", t) for t in list(ON)[:3]] + [("I", t) for t in list(ON)[3:]]
FN = {"age": "Age", "male": "Male sex", "a_sbp_24h_mean": "Systolic BP", "a_dbp_24h_mean": "Diastolic BP", "pulse_24h_mean": "Heart rate",
      "resp_24h_mean": "Respiratory rate", "spo2_24h_mean": "SpO2", "creatinine_first": "Creatinine", "wbc_first": "WBC", "crp_first": "CRP",
      "hemoglobin_first": "Hemoglobin", "hematocrit_first": "Hematocrit", "platelet_first": "Platelet count", "pt_inr_first": "PT-INR", "aptt_first": "aPTT",
      "urine_output_24h_cumulative": "Urine output (24 h)", "is_elective_surgery": "Elective surgery", "is_emergency_surgery": "Emergency surgery",
      "mech_vent_24h": "Invasive ventilation", "norepinephrine_24h": "Norepinephrine", "crrt_24h": "CRRT (24 h)", "sofa_resp": "SOFA respiratory",
      "sofa_coag": "SOFA coagulation", "sofa_liver": "SOFA liver", "sofa_cardio": "SOFA cardiovascular", "sofa_cns": "SOFA CNS", "sofa_renal": "SOFA renal",
      "sofa_total": "SOFA total", "apache2_score": "APACHE II", "rbc_volume_24h": "RBC volume (24 h)", "ffp_volume_24h": "FFP volume (24 h)",
      "platelet_volume_24h": "Platelet volume (24 h)"}
name = lambda c: FN.get(c, FN.get(c.replace("_missing", ""), c) + " missing" if c.endswith("_missing") else c)
pred = {a: pd.read_parquet(OUT / f"pred_test_{a}_full.parquet") for a in "MI"}

# ---- Fig 2 ROC
fig, axs = plt.subplots(2, 3, figsize=(15, 10)); axs = axs.ravel()
for ax, (a, t) in zip(axs, PANELS):
    y = pred[a][t].values
    for m in MN:
        p = pred[a][f"{t}__{m}"].values; f, s, _ = roc_curve(y, p)
        ax.plot(f, s, color=COL[m], lw=1.8, label=f"{MN[m]} ({roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_title(f"{ON[t]} (n = {len(y):,}; events = {int(y.sum())})", fontsize=11)
    ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity"); ax.legend(title="Model (AUROC)", fontsize=9, loc="lower right")
axs[5].axis("off"); fig.tight_layout(); fig.savefig(REV / "Figure_2_ROC.png", dpi=300); plt.close(fig)

# ---- Fig 3 calibration (CatBoost, deciles)
fig, axs = plt.subplots(2, 3, figsize=(15, 10)); axs = axs.ravel()
for ax, (a, t) in zip(axs, PANELS):
    y = pred[a][t].values; p = pred[a][f"{t}__CatBoost"].values
    q = pd.qcut(p, 10, duplicates="drop"); g = pd.DataFrame({"y": y, "p": p}).groupby(q, observed=True).mean()
    lim = max(g.p.max(), g.y.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8); ax.plot(g.p, g.y, "o-", color="#d62728")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_title(ON[t], fontsize=11); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed proportion")
axs[5].axis("off"); fig.tight_layout(); fig.savefig(REV / "Figure_3_calibration.png", dpi=300); plt.close(fig)

# ---- Fig 4 ablation forest
c = pd.read_csv(OUT / "C_ablation.csv"); c = c[c.Analysis.isin(["M", "I"])]
fig, axs = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
rows = [(t, m) for _, t in PANELS for m in MN]
for ax, (fs, ttl) in zip(axs, (("no_transfusion", "Transfusion volumes removed"), ("no_scores", "SOFA / APACHE II removed"))):
    for i, (t, m) in enumerate(rows):
        r = c[(c.Outcome == t) & (c.Model == m) & (c.ablation == fs)].iloc[0]
        sig = r.ci_lo > 0 or r.ci_hi < 0
        ax.errorbar(r.delta, i, xerr=[[r.delta - r.ci_lo], [r.ci_hi - r.delta]], fmt="o", color=COL[m], mfc=COL[m] if sig else "white", capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    for k in range(1, 5): ax.axhline(k * 5 - 0.5, color="#cccccc", lw=0.6)
    ax.set_title(ttl); ax.set_xlabel("Δ AUROC (full − ablated), 95% CI"); ax.set_xlim(-0.08, 0.1)
axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([f"{ON[t]} – {MN[m]}" for t, m in rows], fontsize=8); axs[0].invert_yaxis()
fig.tight_layout(); fig.savefig(REV / "Figure_4_ablation.png", dpi=300); plt.close(fig)

# ---- Fig 5 grouped SHAP shares (CatBoost): main vs sensitivity
g = pd.read_csv(OUT / "E_grouped_shap.csv"); g = g[g.Model == "CatBoost"]
groups = ["SOFA / APACHE II", "Vital signs / urine output", "Laboratory", "Organ support (MV, norepinephrine, CRRT)", "Age / sex", "Surgery type", "Transfusion volumes"]
gc = ["#4c72b0", "#55a868", "#8172b2", "#937860", "#ccb974", "#8c8c8c", "#c44e52"]
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
for ax, (ans, ttl) in zip(axs, ((["M", "I"], "Main analysis (24-h transfusion volumes)"), (["S"], "Sensitivity analysis (24/48/72-h volumes, 72-h landmark)"))):
    s = g[g.Analysis.isin(ans)].pivot_table(index="Outcome", columns="Group", values="share_pct").loc[list(ON)]
    left = np.zeros(len(s))
    for grp, colr in zip(groups, gc):
        ax.barh([ON[o] for o in s.index], s[grp], left=left, color=colr, label=grp)
        for j, (v, l) in enumerate(zip(s[grp], left)):
            if v >= 6: ax.text(l + v / 2, j, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white")
        left += s[grp].values
    ax.set_title(ttl, fontsize=11); ax.set_xlabel("Share of grouped mean |SHAP| (%)"); ax.invert_yaxis()
axs[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout(); fig.savefig(REV / "Figure_5_grouped_SHAP.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig 6 SHAP beeswarm (CatBoost, main analysis)
fig, axs = plt.subplots(2, 3, figsize=(24, 15)); axs = axs.ravel()
for ax, (a, t) in zip(axs, PANELS):
    X = pd.read_parquet(OUT / f"X_{a}.parquet"); feat = pickle.load(open(OUT / f"models_{a}_full" / "feature_columns.pkl", "rb"))
    Xte = X.loc[X.split == "test", feat]
    mdl = pickle.load(open(OUT / f"models_{a}_full" / f"{t}__CatBoost.pkl", "rb"))
    sv = shap.TreeExplainer(mdl).shap_values(Xte.values)
    ex = shap.Explanation(values=sv, data=Xte.values, feature_names=[name(f) for f in feat])
    plt.sca(ax); shap.plots.beeswarm(ex, max_display=15, show=False, ax=ax, plot_size=None, color_bar=False)
    ax.set_title(ON[t], fontsize=13); ax.set_xlabel("SHAP value (log-odds)")
axs[5].axis("off"); fig.tight_layout(); fig.savefig(REV / "Figure_6_SHAP_beeswarm.png", dpi=200); plt.close(fig)
print("figures written")
