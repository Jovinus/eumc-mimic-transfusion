# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from IPython.display import display

pd.set_option('display.max_columns', None)

# %%
df_orig = pd.read_parquet("../../data/raw/analysis_data.parquet")

# %%
# Define column groups
id_cols = ['subject_id', 'hadm_id']
target_cols = [
    'icu_mortality', 'hospital_mortality', 'mortality_28day',
    'blood_culture_positive_after_72h', 'new_antibiotic_after_72h'
]
categorical_cols = ['surgery_type']
binary_cols = [
    'is_elective_surgery', 'is_emergency_surgery', 'is_non_surgical',
    'mech_vent_24h', 'norepinephrine_24h', 'crrt_initiation'
]

# Numeric features that need median imputation (with missing indicators)
numeric_cols = [
    'a_sbp_24h_mean', 'a_dbp_24h_mean', 'pulse_24h_mean', 'resp_24h_mean', 'spo2_24h_mean',
    'creatinine_first', 'wbc_first', 'crp_first', 'hemoglobin_first', 'hematocrit_first',
    'platelet_first', 'pt_inr_first', 'aptt_first', 'urine_output_24h_cumulative',
    'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal',
    'sofa_total', 'apache2_score'
]

# Transfusion features: missing means no transfusion, impute with 0 (no missing indicator needed)
transfusion_cols = [
    'rbc_volume_24h', 'rbc_volume_48h', 'rbc_volume_72h',
    'ffp_volume_24h', 'ffp_volume_48h', 'ffp_volume_72h',
    'platelet_volume_24h', 'platelet_volume_48h', 'platelet_volume_72h'
]

# %%
# Create composite stratification group variable
# Combines all target outcomes to ensure balanced split across all of them
stratify_cols = [
    'icu_mortality', 'hospital_mortality', 'mortality_28day',
    'blood_culture_positive_after_72h', 'new_antibiotic_after_72h'
]

# Create a string-based group variable combining all target outcomes
df_orig['stratify_group'] = (
    df_orig[stratify_cols]
    .astype(str)
    .agg('_'.join, axis=1)
)

# Handle rare groups: groups with < 20 members are merged into a single 'rare' category
# This ensures sufficient samples for stratified splitting
group_counts = df_orig['stratify_group'].value_counts()
rare_groups = group_counts[group_counts < 20].index.tolist()

if len(rare_groups) > 0:
    n_rare_samples = group_counts[rare_groups].sum()
    print(f"Found {len(rare_groups)} rare groups with < 20 members ({n_rare_samples} total samples). Merging into 'rare' category.")
    df_orig.loc[df_orig['stratify_group'].isin(rare_groups), 'stratify_group'] = 'rare'

print("Stratification group distribution:")
print(df_orig['stratify_group'].value_counts())

# %%
# Train-test split (stratified by composite outcome group)
# Split BEFORE any imputation to avoid data leakage
df_train, df_test = train_test_split(
    df_orig,
    test_size=0.2,
    random_state=1004,
    stratify=df_orig['stratify_group']
)

# Drop the temporary stratify_group column
df_train = df_train.drop(columns=['stratify_group'])
df_test = df_test.drop(columns=['stratify_group'])
df_orig = df_orig.drop(columns=['stratify_group'])

print(f"\nTrain size: {len(df_train)}, Test size: {len(df_test)}")
print("\nOutcome distribution comparison (Train vs Test):")
for col in stratify_cols:
    train_rate = df_train[col].mean()
    test_rate = df_test[col].mean()
    print(f"  {col}: Train={train_rate:.3f}, Test={test_rate:.3f}")

# %%
def create_missing_indicators_and_impute(df_train, df_test, numeric_cols, transfusion_cols):
    """
    Create missing indicators and impute with median values.
    Median is computed from training set only to avoid data leakage.
    Transfusion columns are imputed with 0 (no missing indicator needed).
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training dataframe
    df_test : pd.DataFrame
        Test dataframe
    numeric_cols : list
        List of numeric columns to process with median imputation
    transfusion_cols : list
        List of transfusion columns to impute with 0
        
    Returns:
    --------
    df_train_processed : pd.DataFrame
        Processed training data with missing indicators and imputed values
    df_test_processed : pd.DataFrame
        Processed test data with missing indicators and imputed values
    imputation_stats : dict
        Dictionary containing imputation values used
    """
    df_train_processed = df_train.copy()
    df_test_processed = df_test.copy()
    
    imputation_stats = {}
    
    # Process numeric columns with median imputation and missing indicators
    print("=== Numeric columns (median imputation with missing indicators) ===")
    for col in numeric_cols:
        if col not in df_train.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue
            
        # Create missing indicator column
        missing_col_name = f'{col}_missing'
        df_train_processed[missing_col_name] = df_train[col].isnull().astype(int)
        df_test_processed[missing_col_name] = df_test[col].isnull().astype(int)
        
        # Compute median from training set only
        train_median = df_train[col].median()
        imputation_stats[col] = {'value': train_median, 'method': 'median'}
        
        # Impute missing values with training median
        df_train_processed[col] = df_train[col].fillna(train_median)
        df_test_processed[col] = df_test[col].fillna(train_median)
        
        # Report missing counts
        train_missing = df_train[col].isnull().sum()
        test_missing = df_test[col].isnull().sum()
        if train_missing > 0 or test_missing > 0:
            print(f"{col}: Train missing={train_missing} ({train_missing/len(df_train)*100:.1f}%), "
                  f"Test missing={test_missing} ({test_missing/len(df_test)*100:.1f}%), "
                  f"Median={train_median:.2f}")
    
    # Process transfusion columns with 0 imputation (no missing indicators)
    print("\n=== Transfusion columns (impute with 0, no missing indicator) ===")
    for col in transfusion_cols:
        if col not in df_train.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue
        
        imputation_stats[col] = {'value': 0, 'method': 'zero'}
        
        # Impute missing values with 0
        df_train_processed[col] = df_train[col].fillna(0)
        df_test_processed[col] = df_test[col].fillna(0)
        
        # Report missing counts
        train_missing = df_train[col].isnull().sum()
        test_missing = df_test[col].isnull().sum()
        if train_missing > 0 or test_missing > 0:
            print(f"{col}: Train missing={train_missing} ({train_missing/len(df_train)*100:.1f}%), "
                  f"Test missing={test_missing} ({test_missing/len(df_test)*100:.1f}%), "
                  f"Imputed with 0")
    
    return df_train_processed, df_test_processed, imputation_stats

# %%
# Apply missing indicator creation and imputation
df_train_processed, df_test_processed, imputation_stats = create_missing_indicators_and_impute(
    df_train, df_test, numeric_cols, transfusion_cols
)

# %%
# Verify no missing values remain in numeric and transfusion columns
all_imputed_cols = numeric_cols + transfusion_cols
print("\nVerification - Missing values after imputation:")
print("Train:", df_train_processed[all_imputed_cols].isnull().sum().sum())
print("Test:", df_test_processed[all_imputed_cols].isnull().sum().sum())

# %%
# Display processed data info
print("\nProcessed Train shape:", df_train_processed.shape)
print("Processed Test shape:", df_test_processed.shape)

# Show missing indicator columns
missing_indicator_cols = [col for col in df_train_processed.columns if col.endswith('_missing')]
print(f"\nCreated {len(missing_indicator_cols)} missing indicator columns:")
print(missing_indicator_cols)

# %%
# Save imputation statistics for future use (e.g., inference on new data)
imputation_df = pd.DataFrame.from_dict(imputation_stats, orient='index')
imputation_df.index.name = 'feature'
display(imputation_df)

# %%
# Save processed data
df_train_processed.to_parquet("../../data/processed/train_data.parquet", index=False)
df_test_processed.to_parquet("../../data/processed/test_data.parquet", index=False)
imputation_df.to_csv("../../data/processed/imputation_stats.csv")

print("\nSaved processed data to data/processed/")
# %%
