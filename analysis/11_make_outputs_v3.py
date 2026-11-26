"""Word tables + flow diagram for the v3 re-analysis."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt
from docx.enum.section import WD_ORIENT

BASE = Path(sys.argv[1]); OUT = BASE / "out" / __import__("os").environ.get("RUNDIR", "v3"); REV = BASE / "outputs"; REV.mkdir(exist_ok=True)
ON = {"icu_mortality": "ICU mortality", "hospital_mortality": "In-hospital mortality", "mortality_28day": "28-day mortality",
      "blood_culture_positive_after_72h": "Positive blood culture >72 h", "new_antibiotic_after_72h": "New antibiotic >72 h"}
MN = {"DecisionTree": "Decision tree", "RandomForest": "Random forest", "HistGradientBoosting": "Histogram gradient boosting", "CatBoost": "CatBoost", "ANN": "MLP"}
AN = {"M": "Main (24-h volumes; alive at 24 h)", "I": "Main (24-h volumes; in hospital at 72 h)", "S": "Sensitivity (24/48/72-h volumes; 72-h landmark)"}

doc = Document(); sec = doc.sections[0]; sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = sec.page_height, sec.page_width
doc.styles["Normal"].font.size = Pt(8); doc.styles["Normal"].font.name = "Times New Roman"


def add_table(title, df, note=None):
    p = doc.add_paragraph(); r = p.add_run(title); r.bold = True; r.font.size = Pt(10)
    t = doc.add_table(rows=1, cols=df.shape[1]); t.style = "Table Grid"
    for j, c in enumerate(df.columns):
        t.rows[0].cells[j].text = str(c)
        for run in t.rows[0].cells[j].paragraphs[0].runs: run.bold = True
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row): cells[j].text = "" if pd.isna(v) else str(v)
    if note: doc.add_paragraph(note)
    doc.add_page_break()


for coh, ttl in (("cohort_M", "Table 1. Baseline characteristics: admissions transfused within 24 h of ICU admission and alive at 24 h (main cohort)."),
                 ("cohort_I", "Supplementary Table. Baseline characteristics: main cohort still in hospital at 72 h (cohort for >72-h outcomes)."),
                 ("cohort_S", "Supplementary Table. Baseline characteristics: sensitivity cohort (transfused within 72 h, in hospital at 72 h).")):
    add_table(ttl, pd.read_csv(OUT / f"table1_{coh}.csv"),
              "Values computed before imputation. Unit of analysis: hospital admission; train/test split at the patient level. C-reactive protein missing in >90%.")

b = pd.read_csv(OUT / "B_performance.csv")
b["AUROC (95% CI)"] = b.apply(lambda r: f"{r.AUROC:.3f} ({r.AUROC_lo:.3f}–{r.AUROC_hi:.3f})", axis=1)
for c in ["AUPRC", "Brier", "Calibration slope", "Calibration intercept", "Sensitivity", "Specificity", "PPV", "NPV"]:
    b[c] = b[c].map(lambda v: f"{v:.3f}")
b["Outcome"] = b.Outcome.map(ON); b["Model"] = b.Model.map(MN)
for a in "MIS":
    s = b[b.Analysis == a][["Outcome", "Model", "N_test", "Events", "AUROC (95% CI)", "AUPRC", "Brier", "Calibration slope", "Calibration intercept", "Sensitivity", "Specificity", "PPV", "NPV"]]
    add_table(f"Model performance in the test set. {AN[a]}.", s,
              "95% CI from 1,000 bootstrap resamples. Sensitivity/specificity/PPV/NPV at the Youden threshold derived from out-of-fold training predictions.")

c = pd.read_csv(OUT / "C_ablation.csv")
c["Outcome"] = c.Outcome.map(ON); c["Model"] = c.Model.map(MN)
c["Δ"] = c.apply(lambda r: f"{r.delta:+.3f} ({r.ci_lo:+.3f} to {r.ci_hi:+.3f})", axis=1)
for a in "MIS":
    s = c[c.Analysis == a]
    w = s.pivot(index=["Outcome", "Model"], columns="ablation", values="Δ")
    w.insert(0, "AUROC, full model", s.drop_duplicates(["Outcome", "Model"]).set_index(["Outcome", "Model"]).AUROC_full.map(lambda v: f"{v:.3f}"))
    w = w.rename(columns={"no_transfusion": "Δ AUROC without transfusion volumes (95% CI)", "no_scores": "Δ AUROC without SOFA/APACHE II (95% CI)"})
    order = [(ON[o], m) for o in ON for m in MN.values() if (ON[o], m) in w.index]
    add_table(f"Change in test-set AUROC after removing variable groups. {AN[a]}.", w.loc[order].reset_index(),
              "Δ = full − ablated; Δ>0 means the removed variables improved discrimination. Paired bootstrap, 1,000 resamples.")

g = pd.read_csv(OUT / "E_grouped_shap.csv")
g["Outcome"] = g.Outcome.map(ON); g["Model"] = g.Model.map(MN)
for a in "MIS":
    s = g[g.Analysis == a]
    gp = s.pivot_table(index="Group", columns=["Outcome", "Model"], values="share_pct").round(1)
    gp = gp[[(ON[o], m) for o in ON for m in MN.values() if (ON[o], m) in gp.columns]]
    gp.columns = [f"{o} / {m}" for o, m in gp.columns]
    add_table(f"Share of grouped mean |SHAP| by variable group (%). {AN[a]}.", gp.reset_index(),
              "TreeExplainer, whole test set; SHAP values summed within group per admission before averaging absolute values.")

r = pd.read_csv(OUT / "G_association_logistic.csv")
r = r[r.model.isin(["unadjusted", "adjusted"])]
r["Outcome"] = r.outcome.map(ON); r["p"] = r.p.map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
r["Product (per 500 mL in first 24 h)"] = r.term.map({"rbc_volume_24h": "RBC", "ffp_volume_24h": "FFP", "platelet_volume_24h": "Platelets"})
add_table("Association of 24-h transfusion volume with outcomes: logistic regression, cluster-robust SE by patient.",
          r[["Outcome", "n", "events", "model", "Product (per 500 mL in first 24 h)", "OR (95% CI)", "p"]],
          "Adjusted for age, sex, vital signs, laboratory values, urine output, surgery type, SOFA, APACHE II, mechanical ventilation, norepinephrine and CRRT within 24 h. "
          "Observational association; confounding by indication cannot be excluded.")
doc.save(REV / "Revised_Tables_v3.docx")

fig, ax = plt.subplots(figsize=(10.5, 10)); ax.axis("off"); ax.set_xlim(0, 10.5); ax.set_ylim(0, 11)
def box(x, y, w, h, s):
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=1.8)); ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=10)
def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="k"))
box(0.3, 9.6, 5.4, 1.1, "MIMIC-IV v3.1: hospital admissions with\nan ICU stay (n = 85,242; 65,366 patients)")
box(0.3, 7.4, 5.4, 1.1, "RBC, FFP, or platelet transfusion within 24 h\nof first ICU admission (n = 15,299)")
box(0.3, 5.2, 5.4, 1.1, "Main cohort: alive at 24 h\n(n = 14,947 admissions; 13,291 patients)")
box(0.3, 3.0, 5.4, 1.1, "Cohort for >72-h outcomes: in hospital at 72 h\n(n = 13,524 admissions; 12,124 patients)")
box(6.3, 8.5, 4.0, 1.0, "Excluded: no transfusion within\n24 h (n = 69,943)")
box(6.3, 6.3, 4.0, 1.0, "Excluded: died within 24 h\n(n = 352)")
box(6.3, 4.1, 4.0, 1.0, "Excluded: died or discharged\nwithin 72 h (n = 1,423)")
for y in (9.6, 7.4, 5.2):
    arrow(3.0, y, 3.0, y - 1.1); arrow(3.0, y - 0.6, 6.3, y - 0.6)
box(0.3, 0.6, 5.4, 1.5, "Patient-level split (80/20)\nMain cohort: training 11,965 / test 2,982\n>72-h outcomes: training 10,801 / test 2,723")
arrow(3.0, 3.0, 3.0, 2.1)
fig.savefig(REV / "Figure_1_v3.png", dpi=300, bbox_inches="tight")
print("written", [p.name for p in REV.iterdir()])
