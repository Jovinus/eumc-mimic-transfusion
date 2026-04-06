# %%
"""
Table: Transfusion and Clinical Characteristics by Outcome (ALL PATIENTS)
=========================================================================
Compares transfusion volumes and other characteristics between outcome groups.
Generates tables for each outcome variable, covering all patients in the cohort.

Uses pre-imputation values for accurate statistics.
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
OUTCOMES = [
    'icu_mortality',
    'hospital_mortality', 
    'mortality_28day',
    'blood_culture_positive_after_72h',
    'new_antibiotic_after_72h'
]

OUTCOME_LABELS = {
    'icu_mortality': 'ICU Mortality',
    'hospital_mortality': 'Hospital Mortality',
    'mortality_28day': '28-Day Mortality',
    'blood_culture_positive_after_72h': 'Blood Culture Positive (>72h)',
    'new_antibiotic_after_72h': 'New Antibiotic (>72h)'
}

OUTCOME_GROUP_LABELS = {
    'icu_mortality': {'0': 'Survived', '1': 'Died'},
    'hospital_mortality': {'0': 'Survived', '1': 'Died'},
    'mortality_28day': {'0': 'Survived', '1': 'Died'},
    'blood_culture_positive_after_72h': {'0': 'Negative', '1': 'Positive'},
    'new_antibiotic_after_72h': {'0': 'No', '1': 'Yes'}
}

TRANSFUSION_TYPES = ['rbc_volume', 'ffp_volume', 'platelet_volume']
TIMEFRAMES = ['24h', '48h', '72h']

output_dir = Path("../../result/tables")
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load raw data (before imputation) - ALL PATIENTS
print("Loading raw data...")
df_orig = pd.read_parquet("../../data/raw/analysis_data.parquet")
print(f"Loaded {len(df_orig)} records (ALL patients - full cohort)")

# %%
# Define variable lists

# Base continuous variables (without transfusion)
continuous_vars_base = [
    # Demographics / Severity
    'sofa_total', 'apache2_score',
    # Vital signs
    'a_sbp_24h_mean', 'a_dbp_24h_mean', 'pulse_24h_mean', 'resp_24h_mean', 'spo2_24h_mean',
    # Lab values
    'creatinine_first', 'wbc_first', 'crp_first', 'hemoglobin_first', 'hematocrit_first',
    'platelet_first', 'pt_inr_first', 'aptt_first',
    # Other
    'urine_output_24h_cumulative',
]

# Base categorical variables (without transfusion)
categorical_vars_base = [
    'surgery_type',
    'mech_vent_24h', 'norepinephrine_24h', 'crrt_initiation',
]

# Variable labels
labels = {
    # Severity scores
    'sofa_total': 'SOFA Score',
    'apache2_score': 'APACHE II Score',
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
    # Surgery type
    'surgery_type': 'Surgery Type',
    # Interventions
    'mech_vent_24h': 'Mechanical Ventilation (24h)',
    'norepinephrine_24h': 'Norepinephrine (24h)',
    'crrt_initiation': 'CRRT Initiation',
}


# %%
def create_table_for_timeframe_and_outcome(df_orig, timeframe, outcome):
    """
    Create a comparison table for a specific timeframe and outcome,
    for ALL patients.
    """
    df = df_orig.copy()
    
    # Create outcome group variable
    group_labels = OUTCOME_GROUP_LABELS[outcome]
    df['outcome_group'] = df[outcome].astype(str).map(group_labels)
    df = df.dropna(subset=['outcome_group'])
    
    # Build variable lists for this timeframe
    continuous_vars = continuous_vars_base.copy()
    categorical_vars = categorical_vars_base.copy()
    
    # Add transfusion volume variables for this timeframe
    transfusion_volume_vars = [f"{t}_{timeframe}" for t in TRANSFUSION_TYPES]
    
    # Create transfusion binary and amount variables
    for var in transfusion_volume_vars:
        if var not in df.columns:
            continue
        
        # Binary (Yes/No) - who got transfusion
        binary_name = var.replace('_volume_', '_given_')
        df[binary_name] = (df[var].fillna(0) > 0).map({True: 'Yes', False: 'No'})
        categorical_vars.append(binary_name)
        
        # Volume among recipients
        amount_name = var.replace('_volume_', '_amount_')
        df[amount_name] = df[var].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan)
        continuous_vars.append(amount_name)
        
        # Add labels
        trans_type = var.split('_')[0].upper()
        labels[binary_name] = f'{trans_type} Transfusion ({timeframe})'
        labels[amount_name] = f'{trans_type} Volume ({timeframe}), mL*'
    
    # Convert binary intervention vars to Yes/No
    for var in ['mech_vent_24h', 'norepinephrine_24h', 'crrt_initiation']:
        if var in df.columns:
            df[var] = df[var].map({1: 'Yes', 0: 'No', True: 'Yes', False: 'No'})
    
    # Filter to existing columns
    columns = continuous_vars + categorical_vars
    columns = [col for col in columns if col in df.columns]
    cat_vars = [col for col in categorical_vars if col in df.columns]
    
    # Order for categorical variables
    order = {
        'surgery_type': ['elective_surgery', 'emergency_surgery', 'non_surgical'],
        'mech_vent_24h': ['Yes'],
        'norepinephrine_24h': ['Yes'],
        'crrt_initiation': ['Yes'],
    }
    
    # Add transfusion binary vars to order
    for var in transfusion_volume_vars:
        binary_name = var.replace('_volume_', '_given_')
        if binary_name in cat_vars:
            order[binary_name] = ['Yes']
    
    # Create TableOne
    table = TableOne(
        df,
        columns=columns,
        categorical=cat_vars,
        groupby='outcome_group',
        rename=labels,
        order=order,
        pval=True,
        missing=False,
        overall=True,
        label_suffix=False,
    )
    
    return table, df


def create_combined_table_for_outcome(df_orig, outcome):
    """
    Create a combined comparison table including all timeframes (24h, 48h, 72h)
    for a specific outcome, for ALL patients.
    """
    df = df_orig.copy()
    
    # Create outcome group variable
    group_labels = OUTCOME_GROUP_LABELS[outcome]
    df['outcome_group'] = df[outcome].astype(str).map(group_labels)
    df = df.dropna(subset=['outcome_group'])
    
    # Build variable lists
    continuous_vars = continuous_vars_base.copy()
    categorical_vars = categorical_vars_base.copy()
    
    # Order for categorical variables
    order = {
        'surgery_type': ['elective_surgery', 'emergency_surgery', 'non_surgical'],
        'mech_vent_24h': ['Yes'],
        'norepinephrine_24h': ['Yes'],
        'crrt_initiation': ['Yes'],
    }
    
    # Add transfusion volume variables for ALL timeframes
    for timeframe in TIMEFRAMES:
        transfusion_volume_vars = [f"{t}_{timeframe}" for t in TRANSFUSION_TYPES]
        
        for var in transfusion_volume_vars:
            if var not in df.columns:
                continue
            
            # Binary (Yes/No) - who got transfusion
            binary_name = var.replace('_volume_', '_given_')
            df[binary_name] = (df[var].fillna(0) > 0).map({True: 'Yes', False: 'No'})
            categorical_vars.append(binary_name)
            
            # Volume among recipients
            amount_name = var.replace('_volume_', '_amount_')
            df[amount_name] = df[var].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan)
            continuous_vars.append(amount_name)
            
            # Add labels
            trans_type = var.split('_')[0].upper()
            labels[binary_name] = f'{trans_type} Transfusion ({timeframe})'
            labels[amount_name] = f'{trans_type} Volume ({timeframe}), mL*'
            
            # Add to order
            order[binary_name] = ['Yes']
    
    # Convert binary intervention vars to Yes/No
    for var in ['mech_vent_24h', 'norepinephrine_24h', 'crrt_initiation']:
        if var in df.columns:
            df[var] = df[var].map({1: 'Yes', 0: 'No', True: 'Yes', False: 'No'})
    
    # Filter to existing columns
    columns = continuous_vars + categorical_vars
    columns = [col for col in columns if col in df.columns]
    cat_vars = [col for col in categorical_vars if col in df.columns]
    
    # Create TableOne
    table = TableOne(
        df,
        columns=columns,
        categorical=cat_vars,
        groupby='outcome_group',
        rename=labels,
        order=order,
        pval=True,
        missing=False,
        overall=True,
        label_suffix=False,
    )
    
    return table, df


# %%
# Generate tables for each timeframe
print("\n" + "="*80)
print("Generating Tables by Outcome (ALL PATIENTS)")
print("="*80)

for timeframe in TIMEFRAMES:
    print(f"\n{'='*60}")
    print(f"Timeframe: {timeframe}")
    print(f"{'='*60}")
    print(f"Total patients: {len(df_orig)}")
    
    all_tables = {}
    
    for outcome in OUTCOMES:
        print(f"\n  Processing: {OUTCOME_LABELS[outcome]}")
        
        table, df_outcome = create_table_for_timeframe_and_outcome(df_orig, timeframe, outcome)
        all_tables[outcome] = table
        
        # Get group counts
        group_labels = OUTCOME_GROUP_LABELS[outcome]
        group_counts = df_outcome['outcome_group'].value_counts()
        print(f"    {list(group_labels.values())[0]}: {group_counts.get(list(group_labels.values())[0], 0)}")
        print(f"    {list(group_labels.values())[1]}: {group_counts.get(list(group_labels.values())[1], 0)}")
        
        # Save individual tables
        outcome_name = outcome.replace('_after_72h', '')
        
        csv_path = output_dir / f"table_by_{outcome_name}_{timeframe}.csv"
        table.to_csv(str(csv_path))
        
        html_path = output_dir / f"table_by_{outcome_name}_{timeframe}.html"
        table.to_html(str(html_path))
    
    # Create combined Excel for this timeframe
    excel_path = output_dir / f"tables_by_outcome_{timeframe}.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for outcome, table in all_tables.items():
            sheet_name = OUTCOME_LABELS[outcome][:31]
            table.tableone.to_excel(writer, sheet_name=sheet_name)
    
    print(f"\n  Excel saved: {excel_path.name}")

# %%
# Generate COMBINED tables with all timeframes (24h, 48h, 72h) together
print("\n" + "="*80)
print("Generating COMBINED Tables (All Timeframes: 24h, 48h, 72h)")
print("="*80)
print(f"\nTotal patients: {len(df_orig)}")

combined_tables = {}

for outcome in OUTCOMES:
    print(f"\n  Processing: {OUTCOME_LABELS[outcome]}")
    
    table, df_outcome = create_combined_table_for_outcome(df_orig, outcome)
    combined_tables[outcome] = table
    
    # Get group counts
    group_labels = OUTCOME_GROUP_LABELS[outcome]
    group_counts = df_outcome['outcome_group'].value_counts()
    print(f"    {list(group_labels.values())[0]}: {group_counts.get(list(group_labels.values())[0], 0)}")
    print(f"    {list(group_labels.values())[1]}: {group_counts.get(list(group_labels.values())[1], 0)}")
    
    # Save individual combined tables
    outcome_name = outcome.replace('_after_72h', '')
    
    csv_path = output_dir / f"table_by_{outcome_name}_combined.csv"
    table.to_csv(str(csv_path))
    
    html_path = output_dir / f"table_by_{outcome_name}_combined.html"
    table.to_html(str(html_path))

# Create combined Excel with all outcomes
excel_path = output_dir / "tables_by_outcome_combined.xlsx"

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for outcome, table in combined_tables.items():
        sheet_name = OUTCOME_LABELS[outcome][:31]
        table.tableone.to_excel(writer, sheet_name=sheet_name)

print(f"\n  Combined Excel saved: {excel_path.name}")


# %%
# Create outcome prevalence summary table
print("\n" + "="*80)
print("Outcome Prevalence Summary (ALL PATIENTS)")
print("="*80)

prevalence_data = []
for outcome in OUTCOMES:
    total = len(df_orig)
    positive = df_orig[outcome].sum()
    negative = total - positive
    
    prevalence_data.append({
        'Outcome': OUTCOME_LABELS[outcome],
        'Positive n': int(positive),
        'Positive %': f"{positive/total*100:.1f}%",
        'Negative n': int(negative),
        'Negative %': f"{negative/total*100:.1f}%",
        'Total': total
    })

prevalence_df = pd.DataFrame(prevalence_data)
print(prevalence_df.to_string(index=False))

# Save prevalence table
prevalence_path = output_dir / "outcome_prevalence.csv"
prevalence_df.to_csv(prevalence_path, index=False)
print(f"\nPrevalence table saved: {prevalence_path}")

# %%
# Create transfusion summary by outcome (transfusion rates for all patients)
print("\n" + "="*80)
print("Transfusion Rates by Outcome (ALL PATIENTS)")
print("="*80)

transfusion_summary = []

for outcome in OUTCOMES:
    outcome_label = OUTCOME_LABELS[outcome]
    group_labels = OUTCOME_GROUP_LABELS[outcome]
    
    for timeframe in TIMEFRAMES:
        for trans_type in ['rbc', 'ffp', 'platelet']:
            var = f'{trans_type}_volume_{timeframe}'
            
            if var not in df_orig.columns:
                continue
            
            # Overall
            total = len(df_orig)
            any_transfusion = (df_orig[var].fillna(0) > 0).sum()
            
            # By outcome
            positive_mask = df_orig[outcome] == 1
            negative_mask = df_orig[outcome] == 0
            
            pos_total = positive_mask.sum()
            neg_total = negative_mask.sum()
            
            pos_trans = ((df_orig[var].fillna(0) > 0) & positive_mask).sum()
            neg_trans = ((df_orig[var].fillna(0) > 0) & negative_mask).sum()
            
            transfusion_summary.append({
                'Outcome': outcome_label,
                'Transfusion Type': trans_type.upper(),
                'Timeframe': timeframe,
                f'{group_labels["1"]} n/N': f"{pos_trans}/{pos_total}",
                f'{group_labels["1"]} %': f"{pos_trans/pos_total*100:.1f}%" if pos_total > 0 else "N/A",
                f'{group_labels["0"]} n/N': f"{neg_trans}/{neg_total}",
                f'{group_labels["0"]} %': f"{neg_trans/neg_total*100:.1f}%" if neg_total > 0 else "N/A",
                'Overall %': f"{any_transfusion/total*100:.1f}%"
            })

transfusion_summary_df = pd.DataFrame(transfusion_summary)

# Save transfusion summary
transfusion_summary_path = output_dir / "transfusion_rates_by_outcome.csv"
transfusion_summary_df.to_csv(transfusion_summary_path, index=False)
print(f"Transfusion rates saved: {transfusion_summary_path}")

# Display subset
print("\nTransfusion rates (24h) by ICU Mortality:")
subset = transfusion_summary_df[
    (transfusion_summary_df['Outcome'] == 'ICU Mortality') & 
    (transfusion_summary_df['Timeframe'] == '24h')
]
print(subset.to_string(index=False))

# %%
# Display one example table (combined)
print("\n" + "="*80)
print(f"Example: Combined Table by ICU Mortality (All Timeframes, ALL PATIENTS)")
print("="*80)
print("\nContinuous variables: Mean ± SD")
print("Categorical/Binary variables: n (%)")
print("*Transfusion volumes are reported among recipients only")
print("Includes transfusion data for 24h, 48h, and 72h timeframes")
print("-"*80)

example_table, _ = create_combined_table_for_outcome(df_orig, 'icu_mortality')
print(example_table.tabulate(tablefmt="fancy_grid"))

# %%
print("\n" + "="*80)
print("TABLE GENERATION COMPLETE (ALL PATIENTS)!")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print(f"\nGenerated files:")
print(f"  Per-timeframe tables:")
print(f"    - table_by_{{outcome}}_{{timeframe}}.csv")
print(f"    - table_by_{{outcome}}_{{timeframe}}.html")
print(f"    - tables_by_outcome_{{timeframe}}.xlsx")
print(f"  Combined tables (all timeframes):")
print(f"    - table_by_{{outcome}}_combined.csv")
print(f"    - table_by_{{outcome}}_combined.html")
print(f"    - tables_by_outcome_combined.xlsx")
print(f"  Summary files:")
print(f"    - outcome_prevalence.csv")
print(f"    - transfusion_rates_by_outcome.csv")


# %%
