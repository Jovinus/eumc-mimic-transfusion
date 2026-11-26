"""Build the re-analysis dataset: v2 (corrected) + age, sex, CRRT started within 24 h, timing flags,
patient-level train/test split. Prints aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path(sys.argv[1])
PKG = BASE / "extraction"
P = PKG / "data/mimic-iv-parquet"
OUT = BASE / "out" / "v3"
OUT.mkdir(parents=True, exist_ok=True)
RS = 1004
TARGETS = ["icu_mortality", "hospital_mortality", "mortality_28day", "blood_culture_positive_after_72h", "new_antibiotic_after_72h"]
TF = [f"{p}_volume_{t}" for p in ("rbc", "ffp", "platelet") for t in ("24h", "48h", "72h")]

d = pd.read_parquet(PKG / "reference_output/v2_corrected/analysis_data.parquet")
d["hadm_id"] = d.hadm_id.astype(str); d["subject_id"] = d.subject_id.astype(str)

icu = pd.read_parquet(P / "icu/icustays.parquet")
icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
t0 = icu.groupby("hadm_id").intime.min().rename("t0")
adm = pd.read_parquet(P / "hosp/admissions.parquet", columns=["hadm_id", "admittime", "dischtime", "deathtime"]).set_index("hadm_id")
for c in adm.columns:
    adm[c] = pd.to_datetime(adm[c])
pat = pd.read_parquet(P / "hosp/patients.parquet", columns=["subject_id", "gender", "anchor_age", "anchor_year"]).set_index("subject_id")

d = d.join(t0, on="hadm_id").join(adm, on="hadm_id").join(pat, on="subject_id")
d["age"] = d.anchor_age.astype(float) + (d.admittime.dt.year - d.anchor_year.astype(float))
d["male"] = (d.gender == "M").astype(int)
d["h_death"] = (d.deathtime - d.t0).dt.total_seconds() / 3600
d["h_disch"] = (d.dischtime - d.t0).dt.total_seconds() / 3600

# ---- CRRT with timing (same itemids as v2: 225802, 225803 procedureevents; 227536, 227525 inputevents)
ids = ["225802", "225803", "227536", "227525"]
ev = []
for tab in ("procedureevents", "inputevents", "outputevents"):
    t = pq.read_table(P / f"icu/{tab}.parquet", filters=[("itemid", "in", ids)]).to_pandas()
    tc = "starttime" if "starttime" in t.columns else "charttime"
    print(f"CRRT rows in {tab}: {len(t)}")
    if len(t):
        ev.append(pd.DataFrame({"stay_id": t.stay_id, "time": pd.to_datetime(t[tc])}))
ev = pd.concat(ev).merge(icu[["stay_id", "hadm_id"]], on="stay_id").join(t0, on="hadm_id")
ev["h"] = (ev.time - ev.t0).dt.total_seconds() / 3600
first = ev.groupby("hadm_id").h.min()
d["crrt_first_h"] = d.hadm_id.map(first)
d["crrt_any_recomputed"] = d.crrt_first_h.notna().astype(int)
d["crrt_24h"] = (d.crrt_first_h <= 24).astype(int)
print("recomputed any-time CRRT equals v2 crrt_initiation:", bool((d.crrt_any_recomputed == d.crrt_initiation).all()),
      "| n any", int(d.crrt_initiation.sum()), "| n within 24h", int(d.crrt_24h.sum()), "| negative-time events", int((first < 0).sum()))

# ---- cohorts
tf24 = (d[[c for c in TF if c.endswith("24h")]].fillna(0) > 0).any(axis=1)
tf72 = (d[TF].fillna(0) > 0).any(axis=1)
dead24 = d.h_death <= 24
inhosp72 = d.h_disch > 72
d["cohort_M"] = (tf24 & ~dead24).astype(int)            # main, mortality outcomes
d["cohort_I"] = (tf24 & ~dead24 & inhosp72).astype(int)  # main, >72h outcomes
d["cohort_S"] = (tf72 & inhosp72).astype(int)            # sensitivity: 72h volumes, 72h landmark
print("\nall ICU admissions", len(d), "| transfused <=72h", int(tf72.sum()), "| transfused <=24h", int(tf24.sum()),
      "| of those died <=24h", int((tf24 & dead24).sum()))
for c in ("cohort_M", "cohort_I", "cohort_S"):
    s = d[d[c] == 1]
    print(f"{c}: admissions {len(s)}, patients {s.subject_id.nunique()}")

# ---- patient-level split on the union of cohorts
u = d[(d.cohort_M == 1) | (d.cohort_S == 1)].copy()
y = (u[TARGETS].sum(axis=1) > 0).astype(int)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
tr_idx, te_idx = next(sgkf.split(u, y, groups=u.subject_id))
u["split"] = "train"; u.iloc[te_idx, u.columns.get_loc("split")] = "test"
assert not set(u[u.split == "train"].subject_id) & set(u[u.split == "test"].subject_id)
print("\nunion", len(u), "train", int((u.split == "train").sum()), "test", int((u.split == "test").sum()), "(no patient in both)")
for c, ts in (("cohort_M", TARGETS[:3]), ("cohort_I", TARGETS[3:]), ("cohort_S", TARGETS)):
    s = u[u[c] == 1]
    print(f"\n{c}: train {int((s.split=='train').sum())} test {int((s.split=='test').sum())}")
    for t in ts:
        print(f"   {t:34s} all {int(s[t].sum())} ({100*s[t].mean():.1f}%) | train {100*s[s.split=='train'][t].mean():.1f}% | test {100*s[s.split=='test'][t].mean():.1f}%")
s = u[u.cohort_M == 1]
print("\ncohort_M: age %.1f±%.1f, male %d (%.1f%%), CRRT<=24h %d (%.1f%%), CRRT any-time %d (%.1f%%)" % (
    s.age.mean(), s.age.std(), s.male.sum(), 100 * s.male.mean(), s.crrt_24h.sum(), 100 * s.crrt_24h.mean(), s.crrt_initiation.sum(), 100 * s.crrt_initiation.mean()))
u.drop(columns=["t0", "admittime", "dischtime", "deathtime", "gender", "anchor_age", "anchor_year", "crrt_any_recomputed"]).to_parquet(OUT / "analysis_v3.parquet", index=False)
print("saved", OUT / "analysis_v3.parquet", u.shape)
