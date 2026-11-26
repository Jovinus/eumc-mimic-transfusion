"""Association analysis on the corrected dataset: multivariable logistic regression,
cluster-robust SE by patient. Exposure = RBC / FFP / platelet volume in the first 24 h.
Aggregates only."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
OUT = Path(sys.argv[1]) / "out" / __import__("os").environ.get("RUNDIR", "v3")
d = pd.read_parquet(OUT / "analysis_v3.parquet")
MORT = ["icu_mortality", "hospital_mortality", "mortality_28day"]
INF = ["blood_culture_positive_after_72h", "new_antibiotic_after_72h"]
COV = ["age", "male", "a_sbp_24h_mean", "a_dbp_24h_mean", "pulse_24h_mean", "resp_24h_mean", "spo2_24h_mean",
       "creatinine_first", "wbc_first", "hemoglobin_first", "platelet_first", "pt_inr_first", "aptt_first",
       "urine_output_24h_cumulative", "is_elective_surgery", "is_emergency_surgery", "sofa_total", "apache2_score",
       "mech_vent_24h", "norepinephrine_24h", "crrt_24h"]
EXP = ["rbc_volume_24h", "ffp_volume_24h", "platelet_volume_24h"]
rows = []


def fit(s, y, cols, label, model):
    X = s[cols].astype(float).copy()
    X = X.fillna(X.median())
    X = sm.add_constant(X)
    r = sm.Logit(s[y].astype(float), X).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": pd.factorize(s.subject_id)[0]})
    ci = r.conf_int()
    for c in cols:
        if c.startswith(("rbc", "ffp", "platelet_volume", "rbc_cat")) or model == "adjusted_all":
            rows.append({"cohort": label, "outcome": y, "model": model, "term": c, "n": len(s), "events": int(s[y].sum()),
                         "OR": np.exp(r.params[c]), "lo": np.exp(ci.loc[c, 0]), "hi": np.exp(ci.loc[c, 1]), "p": r.pvalues[c]})


for coh, label, ys in (("cohort_M", "M", MORT), ("cohort_I", "I", INF)):
    s = d[d[coh] == 1].copy()
    for e in EXP:
        s[e] = s[e].fillna(0) / 500.0          # per 500 mL
    s["age"] = s.age / 10; s["urine_output_24h_cumulative"] = s.urine_output_24h_cumulative / 1000
    s["a_sbp_24h_mean"] /= 10; s["a_dbp_24h_mean"] /= 10; s["pulse_24h_mean"] /= 10; s["platelet_first"] /= 50
    # RBC categories (mL): 0, 1-350, 351-700, 701-1400, >1400
    cat = pd.cut(s.rbc_volume_24h * 500, [-1, 0, 350, 700, 1400, 1e9], labels=["0", "1-350", "351-700", "701-1400", ">1400"])
    print(label, "RBC category counts:", cat.value_counts().sort_index().to_dict())
    dm = pd.get_dummies(cat, prefix="rbc_cat").astype(float)
    ref = "rbc_cat_1-350"
    catcols = [c for c in dm.columns if c != ref]
    s = pd.concat([s, dm], axis=1)
    for y in ys:
        fit(s, y, EXP, label, "unadjusted")
        fit(s, y, EXP + COV, label, "adjusted")
        fit(s, y, catcols + ["ffp_volume_24h", "platelet_volume_24h"] + COV, label, "adjusted_rbc_categories(ref 1-350 mL)")
    if label == "M":
        fit(s, "hospital_mortality", EXP + COV, label, "adjusted_all")

r = pd.DataFrame(rows)
r["OR (95% CI)"] = r.apply(lambda x: f"{x.OR:.2f} ({x.lo:.2f}-{x.hi:.2f})", axis=1)
r.to_csv(OUT / "G_association_logistic.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(r[["cohort", "outcome", "model", "term", "n", "events", "OR (95% CI)", "p"]].to_string(index=False))
