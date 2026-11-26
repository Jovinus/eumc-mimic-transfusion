"""v4 = v3 with two definitions corrected:
  mech_vent_24h                     -> invasive mechanical ventilation within 24 h (mimic-code ventilation.sql logic)
  blood_culture_positive_after_72h  -> blood-culture bottles only, 'CANCELLED' excluded, skin commensals require >=2 specimens
Aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(sys.argv[1])
P = BASE / "extraction/data/mimic-iv-parquet"
V3 = BASE / "out/v3"; V4 = BASE / "out/v4"; V4.mkdir(exist_ok=True)
d = pd.read_parquet(V3 / "analysis_v3.parquet")
icu = pd.read_parquet(P / "icu/icustays.parquet", columns=["stay_id", "hadm_id", "intime"])
icu["intime"] = pd.to_datetime(icu.intime)
t0 = icu.groupby("hadm_id").intime.min().rename("t0")

# ---------- invasive ventilation
ev = pd.read_parquet(V3 / "vent_events.parquet").merge(icu[["stay_id", "hadm_id"]], on="stay_id").join(t0, on="hadm_id")
ev["h"] = (pd.to_datetime(ev.charttime) - ev.t0).dt.total_seconds() / 3600
ev = ev[(ev.h >= 0) & (ev.h <= 24)]
NOT_INV_MODE = {"Standby", "CPAP", "SPONT", "DuoPaP"}            # 223849 values not in mimic-code's invasive list
HAM_INV = {"APRV", "APV (cmv)", "Ambient", "(S) CMV", "P-CMV", "SIMV", "APV (simv)", "P-SIMV", "VS", "ASV"}
inv = (((ev.itemid == "226732") & (ev.value == "Endotracheal tube")) |
       ((ev.itemid == "223849") & ev.value.notna() & ~ev.value.isin(NOT_INV_MODE)) |
       ((ev.itemid == "229314") & ev.value.isin(HAM_INV)))
inv_h = set(ev[inv].hadm_id)
d["mech_vent_24h_v2"] = d.mech_vent_24h
d["mech_vent_24h"] = d.hadm_id.isin(inv_h).astype(int)

# ---------- blood culture
mb = pd.read_parquet(P / "hosp/microbiologyevents.parquet",
                     columns=["subject_id", "hadm_id", "micro_specimen_id", "chartdate", "charttime", "spec_type_desc", "org_name"])
mb = mb[mb.spec_type_desc.isin(["BLOOD CULTURE", "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"]) & mb.org_name.notna() & (mb.org_name != "") & (mb.org_name != "CANCELLED")]
mb = mb.drop_duplicates(["micro_specimen_id", "org_name"])
mb["time"] = pd.to_datetime(mb.charttime).fillna(pd.to_datetime(mb.chartdate))
# hadm_id is often missing in microbiologyevents: the original code's handling is kept (rows need a hadm_id)
print("blood-culture isolates:", len(mb), "| hadm_id missing: %.1f%%" % (100 * mb.hadm_id.isna().mean()))
# recover hadm_id from subject_id + time within [admittime, dischtime] for the study admissions
adm = pd.read_parquet(P / "hosp/admissions.parquet", columns=["subject_id", "hadm_id", "admittime", "dischtime"])
adm = adm[adm.hadm_id.isin(d.hadm_id)]
adm["admittime"] = pd.to_datetime(adm.admittime); adm["dischtime"] = pd.to_datetime(adm.dischtime)
na = mb[mb.hadm_id.isna()].drop(columns="hadm_id").merge(adm, on="subject_id")
na = na[(na.time >= na.admittime) & (na.time <= na.dischtime)].drop(columns=["admittime", "dischtime"])
print("isolates without hadm_id recovered into study admissions:", len(na))
mb = pd.concat([mb[mb.hadm_id.notna()], na], ignore_index=True).join(t0, on="hadm_id")
mb = mb[mb.t0.notna()]
mb["h"] = (mb.time - mb.t0).dt.total_seconds() / 3600
late = mb[mb.h > 72].copy()
COMM = r"COAGULASE NEGATIVE|STAPHYLOCOCCUS EPIDERMIDIS|STAPHYLOCOCCUS HOMINIS|STAPHYLOCOCCUS CAPITIS|STAPHYLOCOCCUS HAEMOLYTICUS|STAPHYLOCOCCUS WARNERI|" \
       r"STAPHYLOCOCCUS SAPROPHYTICUS|STAPHYLOCOCCUS SIMULANS|STAPHYLOCOCCUS COHNII|STAPHYLOCOCCUS AURICULARIS|" \
       r"CORYNEBACTERIUM|DIPHTHEROID|BACILLUS SPECIES; NOT ANTHRACIS|PROPIONIBACTERIUM|CUTIBACTERIUM|MICROCOCCUS|" \
       r"VIRIDANS STREPTOCOCCI|STREPTOCOCCUS MITIS|STREPTOCOCCUS ORALIS|STREPTOCOCCUS SANGUINIS|STREPTOCOCCUS MUTANS|STREPTOCOCCUS GORDONII|" \
       r"STREPTOCOCCUS SALIVARIUS|STREPTOCOCCUS PARASANGUINIS|AEROCOCCUS"
late["commensal"] = late.org_name.str.contains(COMM, regex=True)
print("isolates >72h: %d | commensal %d (%.1f%%)" % (len(late), late.commensal.sum(), 100 * late.commensal.mean()))
print("commensal organism names matched:", late[late.commensal].org_name.value_counts().head(25).to_dict())
path = set(late[~late.commensal].hadm_id)
c = late[late.commensal].copy(); c["day"] = c.time.dt.normalize()
# commensal counts only if the same organism grew in >=2 separate specimens on the same or consecutive calendar days
c = c.sort_values(["hadm_id", "org_name", "time"])
g = c.groupby(["hadm_id", "org_name"])
c["gap"] = (c.day - g.day.shift(1)).dt.days
c["newspec"] = c.micro_specimen_id != g.micro_specimen_id.shift(1)
conf = set(c[(c.gap <= 1) & c.newspec].hadm_id)
d["blood_culture_positive_after_72h_v2"] = d.blood_culture_positive_after_72h
d["blood_culture_positive_after_72h"] = d.hadm_id.isin(path | conf).astype(int)

for coh in ("cohort_M", "cohort_I", "cohort_S"):
    s = d[d[coh] == 1]
    print(f"\n{coh} (n={len(s)})")
    print("  ventilation: v2 %d (%.1f%%) -> invasive %d (%.1f%%); 1->0 %d, 0->1 %d" % (
        s.mech_vent_24h_v2.sum(), 100 * s.mech_vent_24h_v2.mean(), s.mech_vent_24h.sum(), 100 * s.mech_vent_24h.mean(),
        ((s.mech_vent_24h_v2 == 1) & (s.mech_vent_24h == 0)).sum(), ((s.mech_vent_24h_v2 == 0) & (s.mech_vent_24h == 1)).sum()))
    a, b = s.blood_culture_positive_after_72h_v2, s.blood_culture_positive_after_72h
    print("  blood culture >72h: v2 %d (%.1f%%) -> corrected %d (%.1f%%); 1->0 %d, 0->1 %d | by pathogen %d, by repeated commensal only %d" % (
        a.sum(), 100 * a.mean(), b.sum(), 100 * b.mean(), ((a == 1) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum(),
        s.hadm_id.isin(path).sum(), (s.hadm_id.isin(conf) & ~s.hadm_id.isin(path)).sum()))
    for sp in ("train", "test"):
        print(f"    {sp}: events {int(b[s.split == sp].sum())} / {int((s.split == sp).sum())}")
d.to_parquet(V4 / "analysis_v3.parquet", index=False)   # same file name so the v3 scripts run unchanged with RUNDIR=v4
print("saved", V4)
