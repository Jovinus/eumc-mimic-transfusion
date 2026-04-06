# %%
"""
Table One: TRANSFUSED PATIENTS ONLY
====================================
Demographic and clinical characteristics for patients who received transfusion.
Compares Train vs Test sets using PRE-IMPUTATION values.

Patients are included if they received any transfusion at any timeframe (24h, 48h, or 72h).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tableone import TableOne

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)

# %%
# Configuration
TRANSFUSION_TYPES = ['rbc_volume', 'ffp_volume', 'platelet_volume']
TIMEFRAMES = ['24h', '48h', '72h']

output_dir = Path("../../result/tables_transfused")
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load raw data
print("Loading raw data...")
df_orig = pd.read_parquet("../../data/raw/analysis_data.parquet")
print(f"Loaded {len(df_orig)} records")


# %%
def has_any_transfusion(df):
    """Check if patient received any transfusion at any timeframe."""
    transfusion_cols = []
    for tf in TIMEFRAMES:
        for t in TRANSFUSION_TYPES:
            col = f"{t}_{tf}"
            if col in df.columns:
                transfusion_cols.append(col)
    return (df[transfusion_cols].fillna(0) > 0).any(axis=1)


# %%
# Show transfusion statistics
print("\n" + "="*60)
print("Transfusion Statistics (Full Cohort)")
print("="*60)

mask = has_any_transfusion(df_orig)
print(f"Any transfusion (any timeframe): {mask.sum()} / {len(df_orig)} ({mask.mean()*100:.1f}%)")

# %%
# Recreate the exact same train/test split used in preprocessing
stratify_cols = [
    'icu_mortality', 'hospital_mortality', 'mortality_28day',
    'blood_culture_positive_after_72h', 'new_antibiotic_after_72h'
]

df_orig['stratify_group'] = (
    df_orig[stratify_cols]
    .astype(str)
    .agg('_'.join, axis=1)
)

group_counts = df_orig['stratify_group'].value_counts()
rare_groups = group_counts[group_counts < 20].index.tolist()

if len(rare_groups) > 0:
    df_orig.loc[df_orig['stratify_group'].isin(rare_groups), 'stratify_group'] = 'rare'

df_train, df_test = train_test_split(
    df_orig,
    test_size=0.2,
    random_state=1004,
    stratify=df_orig['stratify_group']
)

df_train = df_train.drop(columns=['stratify_group'])
df_test = df_test.drop(columns=['stratify_group'])

print(f"\nFull cohort - Train: {len(df_train)}, Test: {len(df_test)}")


# %%
# Filter to transfused patients
train_mask = has_any_transfusion(df_train)
test_mask = has_any_transfusion(df_test)

df_train_tf = df_train[train_mask].copy()
df_test_tf = df_test[test_mask].copy()

print(f"Transfused patients - Train: {len(df_train_tf)}, Test: {len(df_test_tf)}")

# Add group variable
df_train_tf['group'] = 'Training'
df_test_tf['group'] = 'Validation'

df_combined = pd.concat([df_train_tf, df_test_tf], ignore_index=True)

# %%
# Define variables
continuous_vars = [
    'a_sbp_24h_mean', 'a_dbp_24h_mean', 'pulse_24h_mean', 'resp_24h_mean', 'spo2_24h_mean',
    'creatinine_first', 'wbc_first', 'crp_first', 'hemoglobin_first', 'hematocrit_first',
    'platelet_first', 'pt_inr_first', 'aptt_first',
    'urine_output_24h_cumulative',
    'sofa_total', 'apache2_score',
]

# All transfusion volume variables (all timeframes)
transfusion_volume_vars = []
for tf in TIMEFRAMES:
    for t in TRANSFUSION_TYPES:
        transfusion_volume_vars.append(f"{t}_{tf}")

categorical_vars = [
    'surgery_type',
    'mech_vent_24h', 'norepinephrine_24h', 'crrt_initiation',
    'icu_mortality', 'hospital_mortality', 'mortality_28day',
    'blood_culture_positive_after_72h', 'new_antibiotic_after_72h',
]

# Create binary transfusion and volume among recipients for all timeframes
transfusion_binary_vars = []
for var in transfusion_volume_vars:
    if var not in df_combined.columns:
        continue
        
    binary_name = var.replace('_volume_', '_given_')
    df_combined[binary_name] = (df_combined[var].fillna(0) > 0).map({True: 'Yes', False: 'No'})
    transfusion_binary_vars.append(binary_name)
    categorical_vars.append(binary_name)
    
    amount_name = var.replace('_volume_', '_amount_')
    df_combined[amount_name] = df_combined[var].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan)
    continuous_vars.append(amount_name)

# Convert binary variables to Yes/No
binary_vars = ['mech_vent_24h', 'norepinephrine_24h', 'crrt_initiation',
               'icu_mortality', 'hospital_mortality', 'mortality_28day', 
               'blood_culture_positive_after_72h', 'new_antibiotic_after_72h']

for var in binary_vars:
    if var in df_combined.columns:
        df_combined[var] = df_combined[var].map({1: 'Yes', 0: 'No', True: 'Yes', False: 'No'})

columns = continuous_vars + categorical_vars
columns = [col for col in columns if col in df_combined.columns]
categorical_vars = [col for col in categorical_vars if col in df_combined.columns]

# %%
# Labels
labels = {
    'a_sbp_24h_mean': 'Arterial SBP, mmHg',
    'a_dbp_24h_mean': 'Arterial DBP, mmHg',
    'pulse_24h_mean': 'Heart Rate, bpm',
    'resp_24h_mean': 'Respiratory Rate, /min',
    'spo2_24h_mean': 'SpO2, %',
    'creatinine_first': 'Creatinine, mg/dL',
    'wbc_first': 'WBC, ×10³/µL',
    'crp_first': 'CRP, mg/L',
    'hemoglobin_first': 'Hemoglobin, g/dL',
    'hematocrit_first': 'Hematocrit, %',
    'platelet_first': 'Platelet, ×10³/µL',
    'pt_inr_first': 'PT INR',
    'aptt_first': 'aPTT, sec',
    'urine_output_24h_cumulative': 'Urine Output 24h, mL',
    'sofa_total': 'SOFA Score',
    'apache2_score': 'APACHE II Score',
    'surgery_type': 'Surgery Type',
    'mech_vent_24h': 'Mechanical Ventilation (24h)',
    'norepinephrine_24h': 'Norepinephrine (24h)',
    'crrt_initiation': 'CRRT Initiation',
    # Transfusion - Binary
    'rbc_given_24h': 'RBC Transfusion (24h)',
    'rbc_given_48h': 'RBC Transfusion (48h)',
    'rbc_given_72h': 'RBC Transfusion (72h)',
    'ffp_given_24h': 'FFP Transfusion (24h)',
    'ffp_given_48h': 'FFP Transfusion (48h)',
    'ffp_given_72h': 'FFP Transfusion (72h)',
    'platelet_given_24h': 'Platelet Transfusion (24h)',
    'platelet_given_48h': 'Platelet Transfusion (48h)',
    'platelet_given_72h': 'Platelet Transfusion (72h)',
    # Transfusion - Volume (among recipients)
    'rbc_amount_24h': 'RBC Volume (24h), mL*',
    'rbc_amount_48h': 'RBC Volume (48h), mL*',
    'rbc_amount_72h': 'RBC Volume (72h), mL*',
    'ffp_amount_24h': 'FFP Volume (24h), mL*',
    'ffp_amount_48h': 'FFP Volume (48h), mL*',
    'ffp_amount_72h': 'FFP Volume (72h), mL*',
    'platelet_amount_24h': 'Platelet Volume (24h), mL*',
    'platelet_amount_48h': 'Platelet Volume (48h), mL*',
    'platelet_amount_72h': 'Platelet Volume (72h), mL*',
    # Outcomes
    'icu_mortality': 'ICU Mortality',
    'hospital_mortality': 'Hospital Mortality',
    'mortality_28day': '28-Day Mortality',
    'blood_culture_positive_after_72h': 'Blood Culture Positive (>72h)',
    'new_antibiotic_after_72h': 'New Antibiotic (>72h)',
}

# %%
# Order
order = {
    'surgery_type': ['elective_surgery', 'emergency_surgery', 'non_surgical'],
    'mech_vent_24h': ['Yes'],
    'norepinephrine_24h': ['Yes'],
    'crrt_initiation': ['Yes'],
    # Transfusion binary - show only 'Yes'
    'rbc_given_24h': ['Yes'],
    'rbc_given_48h': ['Yes'],
    'rbc_given_72h': ['Yes'],
    'ffp_given_24h': ['Yes'],
    'ffp_given_48h': ['Yes'],
    'ffp_given_72h': ['Yes'],
    'platelet_given_24h': ['Yes'],
    'platelet_given_48h': ['Yes'],
    'platelet_given_72h': ['Yes'],
    # Outcomes - show only 'Yes'
    'icu_mortality': ['Yes'],
    'hospital_mortality': ['Yes'],
    'mortality_28day': ['Yes'],
    'blood_culture_positive_after_72h': ['Yes'],
    'new_antibiotic_after_72h': ['Yes'],
}

# %%
# Create TableOne
# By default, continuous variables are reported as Mean ± SD
print("\n" + "="*60)
print("Generating Table One for Transfused Patients")
print("="*60)

table1 = TableOne(
    df_combined,
    columns=columns,
    categorical=categorical_vars,
    groupby='group',
    rename=labels,
    order=order,
    pval=True,
    pval_adjust='bonferroni',
    missing=True,
    overall=True,
    label_suffix=False,
)

# %%
# Save table
csv_path = output_dir / "table_one_transfused.csv"
table1.to_csv(str(csv_path))
print(f"Saved: {csv_path}")

html_path = output_dir / "table_one_transfused.html"
table1.to_html(str(html_path))
print(f"Saved: {html_path}")

excel_path = output_dir / "table_one_transfused.xlsx"
table1.to_excel(str(excel_path))
print(f"Saved: {excel_path}")

# %%
# Create comparison table: Full Cohort vs Transfused
print("\n" + "="*60)
print("Comparison: Full Cohort vs Transfused Patients")
print("="*60)

comparison_data = []
transfusion_mask = has_any_transfusion(df_orig)

full_n = len(df_orig)
trans_n = transfusion_mask.sum()

for outcome in ['icu_mortality', 'hospital_mortality', 'mortality_28day', 
                'blood_culture_positive_after_72h', 'new_antibiotic_after_72h']:
    full_prev = df_orig[outcome].mean() * 100
    trans_prev = df_orig[transfusion_mask][outcome].mean() * 100
    
    comparison_data.append({
        'Outcome': outcome,
        'Full Cohort N': full_n,
        'Full Cohort Prevalence (%)': round(full_prev, 1),
        'Transfused N': trans_n,
        'Transfused Prevalence (%)': round(trans_prev, 1),
        'Difference (%)': round(trans_prev - full_prev, 1)
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_path = output_dir / "outcome_prevalence_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
print(f"Saved: {comparison_path}")

print("\nOutcome Prevalence - Full vs Transfused:")
print(comparison_df.to_string(index=False))

# %%
# Display table
print("\n" + "="*80)
print("Table One - Transfused Patients")
print("="*80)
print("\nContinuous variables: Mean ± SD")
print("Categorical/Binary variables: n (%)")
print("*Transfusion volumes are reported among recipients only")
print("-"*80)
print(table1.tabulate(tablefmt="fancy_grid"))

# %%
print("\n" + "="*60)
print("TABLE ONE GENERATION COMPLETE (TRANSFUSED PATIENTS)")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print(f"\nGenerated files:")
print(f"  - table_one_transfused.csv")
print(f"  - table_one_transfused.html")
print(f"  - table_one_transfused.xlsx")
print(f"  - outcome_prevalence_comparison.csv")
print(f"\nTotal transfused patients: {len(df_combined)}")
print(f"  - Training: {len(df_train_tf)}")
print(f"  - Validation: {len(df_test_tf)}")


# %%
