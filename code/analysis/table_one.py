# %%
"""
Table One: Demographic and Clinical Characteristics
Compares Train vs Test sets using PRE-IMPUTATION values

Uses the `tableone` package for standardized Table 1 generation.

This script recreates the exact train/test split from preprocess_input.py
and generates a demographic table with statistics computed BEFORE imputation
to avoid contaminating the statistics with imputed values.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tableone import TableOne

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)

# %%
# Load raw data (before imputation)
df_orig = pd.read_parquet("../../data/raw/analysis_data.parquet")
print(f"Loaded {len(df_orig)} records from raw data")

# %%
# Recreate the exact same train/test split used in preprocessing
# This ensures we compare the same patients

stratify_cols = [
    'icu_mortality', 'hospital_mortality', 'mortality_28day',
    'blood_culture_positive_after_72h', 'new_antibiotic_after_72h'
]

# Create stratification group (same logic as preprocess_input.py)
df_orig['stratify_group'] = (
    df_orig[stratify_cols]
    .astype(str)
    .agg('_'.join, axis=1)
)

# Handle rare groups
group_counts = df_orig['stratify_group'].value_counts()
rare_groups = group_counts[group_counts < 20].index.tolist()

if len(rare_groups) > 0:
    df_orig.loc[df_orig['stratify_group'].isin(rare_groups), 'stratify_group'] = 'rare'

# Perform the same split with same random state
df_train, df_test = train_test_split(
    df_orig,
    test_size=0.2,
    random_state=1004,
    stratify=df_orig['stratify_group']
)

# Clean up
df_train = df_train.drop(columns=['stratify_group'])
df_test = df_test.drop(columns=['stratify_group'])

print(f"Train: {len(df_train)}, Test: {len(df_test)}")

# %%
# Create group variable for train/test comparison
# Note: 'Test' is a reserved keyword in tableone, so we use 'Validation' instead
df_train['group'] = 'Training'
df_test['group'] = 'Validation'

# Combine for TableOne
df_combined = pd.concat([df_train, df_test], ignore_index=True)

# %%
# Define variable lists for TableOne

# Continuous variables (will use median [IQR] for non-normal distributions)
continuous_vars = [
    # Vital signs
    'a_sbp_24h_mean', 'a_dbp_24h_mean', 'pulse_24h_mean', 'resp_24h_mean', 'spo2_24h_mean',
    # Lab values
    'creatinine_first', 'wbc_first', 'crp_first', 'hemoglobin_first', 'hematocrit_first',
    'platelet_first', 'pt_inr_first', 'aptt_first',
    # Other
    'urine_output_24h_cumulative',
    # Severity scores
    'sofa_total', 'apache2_score',
]

# Transfusion volume variables (continuous - amount given)
transfusion_volume_vars = [
    'rbc_volume_24h', 'rbc_volume_48h', 'rbc_volume_72h',
    'ffp_volume_24h', 'ffp_volume_48h', 'ffp_volume_72h',
    'platelet_volume_24h', 'platelet_volume_48h', 'platelet_volume_72h'
]

# Categorical variables (surgery_type covers the elective/emergency/non-surgical split)
categorical_vars = [
    'surgery_type',
    'mech_vent_24h', 'norepinephrine_24h', 'crrt_initiation',
    # Outcomes
    'icu_mortality', 'hospital_mortality', 'mortality_28day',
    'blood_culture_positive_after_72h', 'new_antibiotic_after_72h',
]

# Create binary transfusion variables (any transfusion) with Yes/No labels
# Also create volume variables for patients who received transfusion (among recipients only)
transfusion_binary_vars = []
for var in transfusion_volume_vars:
    # Binary indicator (Yes/No)
    binary_name = var.replace('_volume_', '_given_')
    df_combined[binary_name] = (df_combined[var].fillna(0) > 0).map({True: 'Yes', False: 'No'})
    transfusion_binary_vars.append(binary_name)
    categorical_vars.append(binary_name)
    
    # Volume among recipients (set 0 to NaN so it shows as "among recipients")
    amount_name = var.replace('_volume_', '_amount_')
    df_combined[amount_name] = df_combined[var].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan)
    continuous_vars.append(amount_name)

# Convert other binary variables to Yes/No for cleaner display
binary_intervention_vars = ['mech_vent_24h', 'norepinephrine_24h', 'crrt_initiation']
binary_outcome_vars = ['icu_mortality', 'hospital_mortality', 'mortality_28day', 
                       'blood_culture_positive_after_72h', 'new_antibiotic_after_72h']

for var in binary_intervention_vars + binary_outcome_vars:
    if var in df_combined.columns:
        df_combined[var] = df_combined[var].map({1: 'Yes', 0: 'No'})

# All columns to include in table
columns = continuous_vars + categorical_vars

# Filter to only existing columns
columns = [col for col in columns if col in df_combined.columns]
categorical_vars = [col for col in categorical_vars if col in df_combined.columns]

# %%
# Variable labels for nicer display
labels = {
    # Vital signs
    'a_sbp_24h_mean': 'Arterial SBP, mmHg',
    'a_dbp_24h_mean': 'Arterial DBP, mmHg',
    'pulse_24h_mean': 'Heart Rate, bpm',
    'resp_24h_mean': 'Respiratory Rate, /min',
    'spo2_24h_mean': 'SpO2, %',
    # Lab values
    'creatinine_first': 'Creatinine, mg/dL',
    'wbc_first': 'WBC, ×10³/µL',
    'crp_first': 'CRP, mg/L',
    'hemoglobin_first': 'Hemoglobin, g/dL',
    'hematocrit_first': 'Hematocrit, %',
    'platelet_first': 'Platelet, ×10³/µL',
    'pt_inr_first': 'PT INR',
    'aptt_first': 'aPTT, sec',
    'urine_output_24h_cumulative': 'Urine Output 24h, mL',
    # Severity scores
    'sofa_total': 'SOFA Score',
    'apache2_score': 'APACHE II Score',
    # Surgery type
    'surgery_type': 'Surgery Type',
    # Interventions
    'mech_vent_24h': 'Mechanical Ventilation (24h)',
    'norepinephrine_24h': 'Norepinephrine (24h)',
    'crrt_initiation': 'CRRT Initiation',
    # Transfusion - Binary (Yes/No)
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
# Define order for categorical variables (show only 'Yes' for binary variables)
order = {
    'surgery_type': ['elective_surgery', 'emergency_surgery', 'non_surgical'],
    # Interventions - show only 'Yes'
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
# nonnormal: use median [IQR] for all continuous variables (typical for clinical data)
table1 = TableOne(
    df_combined,
    columns=columns,
    categorical=categorical_vars,
    groupby='group',
    nonnormal=continuous_vars,  # Use median [IQR] for all continuous
    rename=labels,
    order=order,
    pval=True,
    pval_adjust='bonferroni',  # Adjust for multiple comparisons
    missing=True,  # Show missing value counts
    overall=True,  # Show overall column
    label_suffix=False,
)

# %%
# Display table
print("\n" + "="*100)
print("TABLE ONE: Baseline Characteristics (Pre-Imputation Values)")
print("="*100)
print("\nContinuous variables: Median [Q1, Q3]")
print("Categorical/Binary variables: n (%)")
print("P-values: Mann-Whitney U (continuous), Chi-square (categorical), Bonferroni adjusted")
print("*Transfusion volumes are reported among recipients only")
print("-"*100)
print(table1.tabulate(tablefmt="fancy_grid"))

# %%
# Save outputs to result directory
# CSV
csv_path = "../../result/table_one.csv"
table1.to_csv(csv_path)
print(f"\nCSV saved to: {csv_path}")

# Excel
excel_path = "../../result/table_one.xlsx"
table1.to_excel(excel_path)
print(f"Excel saved to: {excel_path}")

# HTML (for easy viewing in browser)
html_path = "../../result/table_one.html"
table1.to_html(html_path)
print(f"HTML saved to: {html_path}")

# %%
# Summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total patients: {len(df_combined)}")
print(f"Training set: {len(df_train)} ({len(df_train)/len(df_combined)*100:.1f}%)")
print(f"Test set: {len(df_test)} ({len(df_test)/len(df_combined)*100:.1f}%)")
print("\nNote: Statistics calculated from PRE-IMPUTATION values")
print("      Missing values are reported in the table")

# %%
# Also print the raw dataframe for inspection
print("\n" + "="*100)
print("TABLE ONE (DataFrame format)")
print("="*100)
print(table1.tableone)
# %%
