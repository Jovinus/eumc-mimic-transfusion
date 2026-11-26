"""Table 1 for the re-analysis cohorts (before imputation). Aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(sys.argv[1]) / "out" / __import__("os").environ.get("RUNDIR", "v3")
d = pd.read_parquet(OUT / "analysis_v3.parquet")
CONT = [("Age, years", "age"), ("Systolic blood pressure, mmHg", "a_sbp_24h_mean"), ("Diastolic blood pressure, mmHg", "a_dbp_24h_mean"),
        ("Heart rate, /min", "pulse_24h_mean"), ("Respiratory rate, /min", "resp_24h_mean"), ("SpO2, %", "spo2_24h_mean"),
        ("Hemoglobin, g/dL", "hemoglobin_first"), ("Hematocrit, %", "hematocrit_first"), ("SOFA score", "sofa_total"), ("APACHE II score", "apache2_score")]
SKEW = [("Creatinine, mg/dL", "creatinine_first"), ("WBC, x10^3/uL", "wbc_first"), ("C-reactive protein, mg/L", "crp_first"),
        ("Platelet count, x10^3/uL", "platelet_first"), ("PT-INR", "pt_inr_first"), ("aPTT, s", "aptt_first"),
        ("Urine output in first 24 h, mL", "urine_output_24h_cumulative")]
BIN = [("Male sex", "male"), ("Elective surgery", "is_elective_surgery"), ("Emergency surgery", "is_emergency_surgery"), ("Non-surgical", "is_non_surgical"),
       ("Mechanical ventilation within 24 h", "mech_vent_24h"), ("Norepinephrine within 24 h", "norepinephrine_24h"), ("CRRT started within 24 h", "crrt_24h")]
OUTC = [("ICU mortality", "icu_mortality"), ("In-hospital mortality", "hospital_mortality"), ("28-day mortality", "mortality_28day"),
        ("Positive blood culture >72 h", "blood_culture_positive_after_72h"), ("New antibiotic >72 h", "new_antibiotic_after_72h")]


def col(s, outs, tfw):
    r = {"Admissions, n": f"{len(s):,}", "Patients, n": f"{s.subject_id.nunique():,}"}
    for lab, c in CONT:
        r[lab + ", mean ± SD"] = f"{s[c].mean():.1f} ± {s[c].std():.1f}"
    for lab, c in SKEW:
        r[lab + ", median [IQR]"] = f"{s[c].median():.1f} [{s[c].quantile(.25):.1f}–{s[c].quantile(.75):.1f}]"
    for lab, c in BIN:
        r[lab + ", n (%)"] = f"{int(s[c].sum()):,} ({100*s[c].mean():.1f})"
    for prod, name in (("rbc", "RBC"), ("ffp", "FFP"), ("platelet", "Platelets")):
        for w in tfw:
            v = s[f"{prod}_volume_{w}"].fillna(0); pos = v[v > 0]
            r[f"{name} within {w[:-1]} h: recipients, n (%)"] = f"{len(pos):,} ({100*len(pos)/len(s):.1f})"
            r[f"{name} within {w[:-1]} h: volume among recipients, mL, median [IQR]"] = f"{pos.median():.0f} [{pos.quantile(.25):.0f}–{pos.quantile(.75):.0f}]"
    for lab, c in OUTC:
        if c in outs:
            r[lab + ", n (%)"] = f"{int(s[c].sum()):,} ({100*s[c].mean():.1f})"
    return r


for coh, outs, tfw in (("cohort_M", [c for _, c in OUTC[:3]], ["24h"]), ("cohort_I", [c for _, c in OUTC[3:]], ["24h"]), ("cohort_S", [c for _, c in OUTC], ["24h", "48h", "72h"])):
    s = d[d[coh] == 1]
    t = pd.DataFrame({"Overall": col(s, outs, tfw), "Training": col(s[s.split == "train"], outs, tfw), "Test": col(s[s.split == "test"], outs, tfw)})
    t.index.name = "Variable"
    t.to_csv(OUT / f"table1_{coh}.csv", encoding="utf-8-sig")
    miss = s[[c for _, c in CONT + SKEW]].isna().mean().mul(100).round(1)
    miss[miss > 0].to_csv(OUT / f"missing_{coh}.csv")
    if coh == "cohort_M":
        pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
        print(t.to_string()); print("\nmissing %:", miss[miss > 0].to_dict())
