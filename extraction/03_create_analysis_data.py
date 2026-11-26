"""
Step 3. Analysis dataset 생성 (selected_features/*.parquet -> analysis/analysis_data.parquet)

Usage:
    python 03_create_analysis_data.py \
        --data-path     <MIMIC-IV parquet dir (contains hosp/, icu/)> \
        --feature-path  <Step 2 output dir> \
        --save-path     <output dir for analysis_data.parquet>

Original: code/eumc/transfusion/generate-cohort/create_analysis_data.py
(handoff 버전은 경로 설정 부분만 CLI 인자로 바꿨고, 처리 로직은 원본과 동일합니다.)
"""
# %%
import argparse
import numpy as np
import os
import pandas as pd

try:
    from IPython.display import display
except ImportError:  # IPython 없이 터미널에서 실행하는 경우
    display = print

pd.set_option("display.max_columns", None)

# %%
# Path configuration
# 기본 경로: 이 스크립트 위치 기준 ../data (어디서 실행해도 패키지 안의 data/ 사용)
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_DATA = os.path.normpath(os.path.join(_HERE, "..", "data"))
_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--data-path", default=os.path.join(_DATA, "mimic-iv-parquet"),
                     help="Step 1에서 만든 MIMIC-IV parquet 디렉토리 (icustays 사용)")
_parser.add_argument("--feature-path", default=os.path.join(_DATA, "selected_features_v2"),
                     help="Step 2 결과 디렉토리")
_parser.add_argument("--save-path", default=os.path.join(_DATA, "analysis_v2"),
                     help="analysis_data.parquet 저장 디렉토리")
_args, _ = _parser.parse_known_args()  # parse_known_args: Jupyter/VSCode 셀 실행 시에도 동작

raw_data_path = _args.data_path
feature_path = _args.feature_path
save_path = _args.save_path
os.makedirs(save_path, exist_ok=True)


# %%
def get_first_icu_intime(raw_data_path: str) -> pd.DataFrame:
    """
    Get first ICU admission time per hospital admission (hadm_id).
    This is used as the reference time for calculating 24h windows.
    """
    icustays = pd.read_parquet(
        os.path.join(raw_data_path, "icu", "icustays.parquet")
    )[["stay_id", "subject_id", "hadm_id", "intime"]]
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    
    # Get first ICU intime per admission
    first_icu_intime = (
        icustays
        .groupby("hadm_id")
        .agg({
            "subject_id": "first",
            "intime": "min"
        })
        .reset_index()
        .rename(columns={"intime": "first_icu_intime"})
    )
    
    return first_icu_intime


# %%
def get_vital_signs_24h_mean(
    feature_path: str,
    first_icu_intime: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate first 24 hours mean values for vital signs.
    
    Features: a_sbp, a_dbp, pulse, resp, spo2
    
    Args:
        feature_path: Path to the selected features parquet files
        first_icu_intime: DataFrame with first ICU intime per hadm_id
        
    Returns:
        DataFrame with first 24h mean for each vital sign, pivoted to wide format
    """
    print("Computing first 24h mean for vital signs...")
    
    # Load vital signs
    df_vital = pd.read_parquet(
        os.path.join(feature_path, "vital_sign.parquet")
    )
    
    # Merge with first ICU intime
    df_vital = df_vital.merge(
        first_icu_intime[["hadm_id", "first_icu_intime"]],
        on="hadm_id",
        how="inner"
    )
    
    # Calculate hours from first ICU admission
    df_vital["hours_from_icu"] = (
        df_vital["charttime"] - df_vital["first_icu_intime"]
    ).dt.total_seconds() / 3600
    
    # Filter first 24 hours
    df_vital_24h = df_vital[
        (df_vital["hours_from_icu"] >= 0) & 
        (df_vital["hours_from_icu"] <= 24)
    ].copy()
    
    # Calculate mean per hadm_id and feature_name
    vital_mean = (
        df_vital_24h
        .groupby(["subject_id", "hadm_id", "feature_name"])["valuenum"]
        .mean()
        .reset_index()
    )
    
    # Pivot to wide format
    vital_wide = vital_mean.pivot_table(
        index=["subject_id", "hadm_id"],
        columns="feature_name",
        values="valuenum"
    ).reset_index()
    
    # Rename columns to indicate 24h mean
    vital_features = ["a_sbp", "a_dbp", "pulse", "resp", "spo2"]
    for col in vital_features:
        if col in vital_wide.columns:
            vital_wide = vital_wide.rename(columns={col: f"{col}_24h_mean"})
    
    # Flatten column names
    vital_wide.columns.name = None
    
    print(f"  Completed: {len(vital_wide)} admissions with vital sign data")
    
    return vital_wide


# %%
def get_laboratory_first_value(
    feature_path: str,
    first_icu_intime: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get first laboratory values within the admission.
    
    Features: creatinine, wbc, crp, hemoglobin, hematocrit, platelet, pt_inr, aptt
    
    Args:
        feature_path: Path to the selected features parquet files
        first_icu_intime: DataFrame with first ICU intime per hadm_id
        
    Returns:
        DataFrame with first value for each lab, pivoted to wide format
    """
    print("Getting first laboratory values...")
    
    # Load laboratory data
    df_lab = pd.read_parquet(
        os.path.join(feature_path, "laboratory.parquet")
    )
    
    # Merge with first ICU intime
    df_lab = df_lab.merge(
        first_icu_intime[["hadm_id", "first_icu_intime"]],
        on="hadm_id",
        how="inner"
    )
    
    # Calculate hours from first ICU admission
    df_lab["hours_from_icu"] = (
        df_lab["charttime"] - df_lab["first_icu_intime"]
    ).dt.total_seconds() / 3600
    
    # Filter to include only values from before or within 24h of ICU admission
    # (to get the "first" value which could be from before ICU or shortly after)
    # Sort by charttime and get first per feature
    df_lab_sorted = df_lab.sort_values("charttime")
    
    # Get first value per hadm_id and feature_name
    lab_first = (
        df_lab_sorted
        .groupby(["subject_id", "hadm_id", "feature_name"])
        .first()
        .reset_index()
        [["subject_id", "hadm_id", "feature_name", "valuenum"]]
    )
    
    # Pivot to wide format
    lab_wide = lab_first.pivot_table(
        index=["subject_id", "hadm_id"],
        columns="feature_name",
        values="valuenum"
    ).reset_index()
    
    # Rename columns to indicate first value
    lab_features = ["creatinine", "wbc", "crp", "hemoglobin", "hematocrit", "platelet", "pt_inr", "aptt"]
    for col in lab_features:
        if col in lab_wide.columns:
            lab_wide = lab_wide.rename(columns={col: f"{col}_first"})
    
    # Flatten column names
    lab_wide.columns.name = None
    
    print(f"  Completed: {len(lab_wide)} admissions with laboratory data")
    
    return lab_wide


# %%
def get_urine_output_24h_cumulative(
    feature_path: str,
    first_icu_intime: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate first 24 hours cumulative urine output.
    
    Args:
        feature_path: Path to the selected features parquet files
        first_icu_intime: DataFrame with first ICU intime per hadm_id
        
    Returns:
        DataFrame with first 24h cumulative urine output
    """
    print("Computing first 24h cumulative urine output...")
    
    # Load urine output data
    df_urine = pd.read_parquet(
        os.path.join(feature_path, "urine_output.parquet")
    )
    
    # Merge with first ICU intime
    df_urine = df_urine.merge(
        first_icu_intime[["hadm_id", "first_icu_intime"]],
        on="hadm_id",
        how="inner"
    )
    
    # Calculate hours from first ICU admission
    df_urine["hours_from_icu"] = (
        df_urine["charttime"] - df_urine["first_icu_intime"]
    ).dt.total_seconds() / 3600
    
    # Filter first 24 hours
    df_urine_24h = df_urine[
        (df_urine["hours_from_icu"] >= 0) & 
        (df_urine["hours_from_icu"] <= 24)
    ].copy()
    
    # Calculate cumulative sum per hadm_id
    urine_cumulative = (
        df_urine_24h
        .groupby(["subject_id", "hadm_id"])["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "urine_output_24h_cumulative"})
    )
    
    print(f"  Completed: {len(urine_cumulative)} admissions with urine output data")
    
    return urine_cumulative


# %%
def load_precomputed_features(feature_path: str) -> dict:
    """
    Load all pre-computed feature files from the selected_features folder.
    
    Returns:
        Dictionary with feature name as key and DataFrame as value
    """
    print("Loading pre-computed features...")
    
    features = {}
    
    # Surgery type
    print("  - Surgery type")
    features["surgery_type"] = pd.read_parquet(
        os.path.join(feature_path, "surgery_type.parquet")
    )
    
    # Severity scores (SOFA and APACHE)
    print("  - Severity scores (SOFA & APACHE)")
    features["severity_scores"] = pd.read_parquet(
        os.path.join(feature_path, "severity_scores.parquet")
    )
    
    # Mechanical ventilation
    print("  - Mechanical ventilation")
    features["ventilation"] = pd.read_parquet(
        os.path.join(feature_path, "ventilation.parquet")
    )
    
    # Vasopressor
    print("  - Vasopressor")
    features["vasopressor"] = pd.read_parquet(
        os.path.join(feature_path, "vasopressor.parquet")
    )
    
    # Transfusion volume
    print("  - Transfusion volume")
    features["transfusion"] = pd.read_parquet(
        os.path.join(feature_path, "transfusion_volume.parquet")
    )
    
    # Mortality
    print("  - Mortality")
    features["mortality"] = pd.read_parquet(
        os.path.join(feature_path, "mortality.parquet")
    )
    
    # Infection
    print("  - Infection")
    features["infection"] = pd.read_parquet(
        os.path.join(feature_path, "infection.parquet")
    )
    
    # CRRT initiation
    print("  - CRRT initiation")
    features["crrt"] = pd.read_parquet(
        os.path.join(feature_path, "crrt_initiation.parquet")
    )
    
    return features


# %%
# Physiologically plausible ranges for data cleaning
# Values outside these ranges are considered errors and set to NaN
VALID_RANGES = {
    # Vital signs (24h mean)
    "a_sbp_24h_mean": (20, 300),       # Systolic BP: 20-300 mmHg
    "a_dbp_24h_mean": (10, 200),       # Diastolic BP: 10-200 mmHg
    "pulse_24h_mean": (20, 300),       # Heart rate: 20-300 bpm
    "resp_24h_mean": (1, 60),          # Respiratory rate: 1-60 /min
    "spo2_24h_mean": (0, 100),         # SpO2: 0-100%
    
    # Laboratory values (first)
    "creatinine_first": (0, 30),       # Creatinine: 0-30 mg/dL
    "wbc_first": (0, 500),             # WBC: 0-500 x10^9/L
    "crp_first": (0, 500),             # CRP: 0-500 mg/L
    "hemoglobin_first": (1, 25),       # Hemoglobin: 1-25 g/dL
    "hematocrit_first": (1, 70),       # Hematocrit: 1-70%
    "platelet_first": (0, 2000),       # Platelets: 0-2000 x10^9/L
    "pt_inr_first": (0.1, 20),         # PT/INR: 0.1-20
    "aptt_first": (0, 200),            # aPTT: 0-200 seconds
    
    # Urine output
    "urine_output_24h_cumulative": (0, 20000),  # Urine: 0-20000 mL/24h
    
    # Severity scores
    "sofa_resp": (0, 4),               # SOFA components: 0-4
    "sofa_coag": (0, 4),
    "sofa_liver": (0, 4),
    "sofa_cardio": (0, 4),
    "sofa_cns": (0, 4),
    "sofa_renal": (0, 4),
    "sofa_total": (0, 24),             # Total SOFA: 0-24
    "apache2_score": (0, 71),          # APACHE II: 0-71
    
    # Transfusion volumes (mL)
    "rbc_volume_24h": (0, 50000),
    "rbc_volume_48h": (0, 100000),
    "rbc_volume_72h": (0, 150000),
    "ffp_volume_24h": (0, 50000),
    "ffp_volume_48h": (0, 100000),
    "ffp_volume_72h": (0, 150000),
    "platelet_volume_24h": (0, 50000),
    "platelet_volume_48h": (0, 100000),
    "platelet_volume_72h": (0, 150000),
}


def clean_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean analysis data by removing physiologically implausible values.
    
    Values outside the defined valid ranges are set to NaN.
    
    Args:
        df: DataFrame with analysis data
        
    Returns:
        Cleaned DataFrame with implausible values set to NaN
    """
    print("\nCleaning data (removing implausible values)...")
    
    df_clean = df.copy()
    total_cleaned = 0
    
    for col, (min_val, max_val) in VALID_RANGES.items():
        if col in df_clean.columns:
            # Count values outside range
            mask_invalid = (df_clean[col] < min_val) | (df_clean[col] > max_val)
            n_invalid = mask_invalid.sum()
            
            if n_invalid > 0:
                # Set invalid values to NaN
                df_clean.loc[mask_invalid, col] = np.nan
                total_cleaned += n_invalid
                print(f"  {col}: {n_invalid} invalid values removed "
                      f"(valid range: {min_val}-{max_val})")
    
    if total_cleaned == 0:
        print("  No invalid values found.")
    else:
        print(f"  Total: {total_cleaned} invalid values removed")
    
    return df_clean


# %%
def create_analysis_data(
    raw_data_path: str,
    feature_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Create analysis data by combining:
    - First 24h mean for vital signs (a_sbp, a_dbp, pulse, resp, spo2)
    - First value for laboratory (creatinine, wbc, crp, hemoglobin, hematocrit, platelet, pt_inr, aptt)
    - First 24h cumulative urine output
    - Surgery type (elective, emergency, non-surgical)
    - Severity scores (SOFA & APACHE II)
    - Mechanical ventilation (24h)
    - Vasopressor (norepinephrine 24h)
    - Transfusion volume (RBC, FFP, Platelet at 24h, 48h, 72h)
    - Mortality (ICU, hospital, 28-day)
    - Infection indicators
    
    Args:
        raw_data_path: Path to raw MIMIC-IV data
        feature_path: Path to selected features parquet files
        save_path: Path to save the analysis data
        
    Returns:
        DataFrame with all analysis features
    """
    print("\n" + "=" * 60)
    print("CREATING ANALYSIS DATA")
    print("=" * 60 + "\n")
    
    # Get first ICU intime per admission
    print("Loading first ICU admission times...")
    first_icu_intime = get_first_icu_intime(raw_data_path)
    print(f"  Total admissions with ICU stays: {len(first_icu_intime)}\n")
    
    # Get vital signs - first 24h mean
    vital_24h_mean = get_vital_signs_24h_mean(feature_path, first_icu_intime)
    
    # Get laboratory - first values
    lab_first = get_laboratory_first_value(feature_path, first_icu_intime)
    
    # Get urine output - first 24h cumulative
    urine_24h = get_urine_output_24h_cumulative(feature_path, first_icu_intime)
    
    # Load pre-computed features
    print()
    precomputed = load_precomputed_features(feature_path)
    
    # Merge all features
    print("\nMerging all features...")
    
    # Start with first_icu_intime as base
    analysis_data = first_icu_intime[["subject_id", "hadm_id"]].copy()
    
    # Merge vital signs
    analysis_data = analysis_data.merge(
        vital_24h_mean,
        on=["subject_id", "hadm_id"],
        how="left"
    )
    
    # Merge laboratory
    analysis_data = analysis_data.merge(
        lab_first,
        on=["subject_id", "hadm_id"],
        how="left"
    )
    
    # Merge urine output
    analysis_data = analysis_data.merge(
        urine_24h,
        on=["subject_id", "hadm_id"],
        how="left"
    )
    
    # Merge surgery type
    surgery_cols = ["subject_id", "hadm_id", "surgery_type", 
                    "is_elective_surgery", "is_emergency_surgery", "is_non_surgical"]
    analysis_data = analysis_data.merge(
        precomputed["surgery_type"][surgery_cols],
        on=["subject_id", "hadm_id"],
        how="left"
    )
    
    # Merge severity scores (SOFA & APACHE)
    severity_cols = ["subject_id", "hadm_id", 
                     "sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", 
                     "sofa_cns", "sofa_renal", "sofa_total", "apache2_score"]
    analysis_data = analysis_data.merge(
        precomputed["severity_scores"][severity_cols],
        on=["subject_id", "hadm_id"],
        how="left"
    )
    
    # Merge ventilation (binary flag for 24h)
    analysis_data = analysis_data.merge(
        precomputed["ventilation"],
        on=["subject_id", "hadm_id"],
        how="left"
    )
    # Fill NaN with 0 (no ventilation)
    analysis_data["mech_vent_24h"] = analysis_data["mech_vent_24h"].fillna(0).astype(int)
    
    # Merge vasopressor (binary flag for norepinephrine 24h)
    analysis_data = analysis_data.merge(
        precomputed["vasopressor"],
        on=["subject_id", "hadm_id"],
        how="left"
    )
    # Fill NaN with 0 (no vasopressor)
    analysis_data["norepinephrine_24h"] = analysis_data["norepinephrine_24h"].fillna(0).astype(int)
    
    # Merge transfusion volume
    analysis_data = analysis_data.merge(
        precomputed["transfusion"],
        on=["subject_id", "hadm_id"],
        how="left"
    )
    # Fill NaN with 0 (no transfusion)
    transfusion_cols = [col for col in precomputed["transfusion"].columns 
                        if "_volume_" in col]
    for col in transfusion_cols:
        if col in analysis_data.columns:
            analysis_data[col] = analysis_data[col].fillna(0)
    
    # Merge mortality
    mortality_cols = ["subject_id", "hadm_id", 
                      "icu_mortality", "hospital_mortality", "mortality_28day"]
    analysis_data = analysis_data.merge(
        precomputed["mortality"][mortality_cols],
        on=["subject_id", "hadm_id"],
        how="left"
    )
    
    # Merge infection
    analysis_data = analysis_data.merge(
        precomputed["infection"],
        on=["subject_id", "hadm_id"],
        how="left"
    )
    
    # Merge CRRT initiation
    analysis_data = analysis_data.merge(
        precomputed["crrt"],
        on=["subject_id", "hadm_id"],
        how="left"
    )
    # Fill NaN with 0 (no CRRT)
    analysis_data["crrt_initiation"] = analysis_data["crrt_initiation"].fillna(0).astype(int)
    
    # Define column order
    id_cols = ["subject_id", "hadm_id"]
    
    vital_cols = [
        "a_sbp_24h_mean", "a_dbp_24h_mean", "pulse_24h_mean", 
        "resp_24h_mean", "spo2_24h_mean"
    ]
    
    lab_cols = [
        "creatinine_first", "wbc_first", "crp_first", "hemoglobin_first",
        "hematocrit_first", "platelet_first", "pt_inr_first", "aptt_first"
    ]
    
    urine_cols = ["urine_output_24h_cumulative"]
    
    surgery_feature_cols = [
        "surgery_type", "is_elective_surgery", "is_emergency_surgery", "is_non_surgical"
    ]
    
    severity_feature_cols = [
        "sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", 
        "sofa_cns", "sofa_renal", "sofa_total", "apache2_score"
    ]
    
    intervention_cols = ["mech_vent_24h", "norepinephrine_24h", "crrt_initiation"]
    
    transfusion_feature_cols = [
        "rbc_volume_24h", "rbc_volume_48h", "rbc_volume_72h",
        "ffp_volume_24h", "ffp_volume_48h", "ffp_volume_72h",
        "platelet_volume_24h", "platelet_volume_48h", "platelet_volume_72h"
    ]
    
    outcome_cols = [
        "icu_mortality", "hospital_mortality", "mortality_28day",
        "blood_culture_positive_after_72h", "new_antibiotic_after_72h"
    ]
    
    # Select only existing columns in order
    all_feature_cols = (
        vital_cols + lab_cols + urine_cols + surgery_feature_cols + 
        severity_feature_cols + intervention_cols + transfusion_feature_cols + 
        outcome_cols
    )
    existing_cols = id_cols + [
        col for col in all_feature_cols if col in analysis_data.columns
    ]
    analysis_data = analysis_data[existing_cols]
    
    # Clean data - remove physiologically implausible values
    analysis_data = clean_analysis_data(analysis_data)
    
    # Save to parquet
    output_file = os.path.join(save_path, "analysis_data.parquet")
    analysis_data.to_parquet(output_file, index=False)
    
    print(f"\nAnalysis data created successfully!")
    print(f"  Total admissions: {len(analysis_data)}")
    print(f"  Total columns: {len(analysis_data.columns)}")
    print(f"  Saved to: {output_file}")
    
    # Display column categories
    print("\nColumns by category:")
    print(f"  ID: {id_cols}")
    print(f"  Vital signs (24h mean): {[c for c in vital_cols if c in analysis_data.columns]}")
    print(f"  Laboratory (first): {[c for c in lab_cols if c in analysis_data.columns]}")
    print(f"  Urine output: {[c for c in urine_cols if c in analysis_data.columns]}")
    print(f"  Surgery type: {[c for c in surgery_feature_cols if c in analysis_data.columns]}")
    print(f"  Severity scores: {[c for c in severity_feature_cols if c in analysis_data.columns]}")
    print(f"  Interventions: {[c for c in intervention_cols if c in analysis_data.columns]}")
    print(f"  Transfusion: {[c for c in transfusion_feature_cols if c in analysis_data.columns]}")
    print(f"  Outcomes: {[c for c in outcome_cols if c in analysis_data.columns]}")
    
    # Display missing data summary
    print("\nMissing data summary:")
    missing = analysis_data.isnull().sum()
    missing_pct = (missing / len(analysis_data) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    })
    print(missing_df[missing_df["Missing Count"] > 0])
    
    return analysis_data


# %%
if __name__ == "__main__":
    analysis_data = create_analysis_data(
        raw_data_path=raw_data_path,
        feature_path=feature_path,
        save_path=save_path
    )
    
    print("\n" + "=" * 60)
    print("DATA PREVIEW")
    print("=" * 60)
    display(analysis_data.head(10))
    
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    display(analysis_data.describe())


# %%
