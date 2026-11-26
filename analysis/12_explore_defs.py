"""Category counts needed to define invasive ventilation and blood-culture contaminants (aggregates only)."""
import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

P = Path(sys.argv[1]) / "extraction/data/mimic-iv-parquet"
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 300)
ce = pq.read_table(P / "icu/chartevents.parquet", columns=["stay_id", "charttime", "itemid", "value"],
                   filters=[("itemid", "in", ["223849", "229314", "226732"])]).to_pandas()
ce.to_parquet(Path(sys.argv[1]) / "out/v3/vent_events.parquet", index=False)
for i, n in (("223849", "Ventilator Mode"), ("229314", "Ventilator Mode (Hamilton)"), ("226732", "O2 Delivery Device(s)")):
    print(f"\n== {i} {n}"); print(ce[ce.itemid == i].value.value_counts(dropna=False).to_string())

mb = pd.read_parquet(P / "hosp/microbiologyevents.parquet")
print("\nmicro columns:", list(mb.columns))
b = mb[mb.spec_type_desc.str.contains("BLOOD", case=False, na=False)]
print("\n== blood specimen types"); print(b.spec_type_desc.value_counts().to_string())
print("\n== org_name in blood specimens (top 80), distinct specimens")
print(b[b.org_name.notna()].drop_duplicates(["micro_specimen_id", "org_name"]).org_name.value_counts().head(80).to_string())
print("charttime missing among blood rows: %.1f%%" % (100 * b.charttime.isna().mean()))
