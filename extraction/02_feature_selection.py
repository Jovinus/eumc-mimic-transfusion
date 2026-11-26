"""
Step 2. Feature extraction (MIMIC-IV parquet -> selected_features/*.parquet)  [v2: 오류 교정판]

v1(원본 재현: 02_feature_selection_v1.py) 대비 교정 사항 (상세: docs/v2_changes.md, 정의: definitions_v2.py)
  [교정 1] 수술 분류: 잘못된 procedureevents itemid(EKG/Blood Cultured/Lumbar Puncture) 제거 -> services 기준만
  [교정 2] GCS: E/V/M 구성요소 최솟값 -> 측정 시점별 총점(E+V+M)의 최솟값 (mimic-code gcs.sql 방식)
  [교정 3] APACHE II 체온: °F/°C 혼재 -> °C로 변환 + 유효범위 (mimic-code vitalsign.sql 방식)
  [교정 4] 항생제: 키워드 부분일치 -> 전신 항균제 성분 목록 + 비전신 투여 제외 + 성분 기준 "새 항생제"
  [교정 5] CRRT: 224270 Dialysis Catheter, 226499 Hemodialysis Output 제거
그 외 모든 로직은 v1과 동일합니다.

Usage:
    python 02_feature_selection.py \
        --data-path  <MIMIC-IV parquet dir (contains hosp/, icu/)> \
        --save-path  <output dir for selected_features>

Original: code/eumc/transfusion/clean/feature_selection.py
(pandas 원본: code/legacy_pandas/02_feature_selection.py)

[DuckDB 고속화]
원본은 chartevents(2.3GB) 같은 대용량 테이블을 `pd.read_parquet()`로 통째로 읽은 뒤
pandas에서 itemid로 필터링했습니다(chartevents만 4회 전체 로드).
이 버전은 DuckDB로 "필요한 컬럼 + 필요한 행(itemid 등)"만 읽고,
그 이후의 형변환, merge, 집계는 원본 pandas 코드를 그대로 사용합니다.
  - 원본 파일의 행 순서를 유지합니다(file_row_number 정렬). 행 순서에 따라 결과가 달라질 수 있는
    정렬/first() 연산이 원본과 동일하게 동작하도록 하기 위함입니다.
  - 집계는 pandas가 하므로 부동소수점 합산 순서도 원본과 같습니다.
"""
# %%
import argparse
import duckdb
import numpy as np
import os
import pandas as pd

import definitions_v2 as defs  # v2 교정 정의 (같은 폴더)

try:
    from IPython.display import display
except ImportError:  # IPython 없이 터미널에서 실행하는 경우
    display = print

pd.set_option("display.max_columns", None)

# %%
# 기본 경로: 이 스크립트 위치 기준 ../data (어디서 실행해도 패키지 안의 data/ 사용)
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_DATA = os.path.normpath(os.path.join(_HERE, "..", "data"))
_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--data-path", default=os.path.join(_DATA, "mimic-iv-parquet"),
                     help="Step 1에서 만든 MIMIC-IV parquet 디렉토리 (hosp/, icu/ 포함)")
_parser.add_argument("--save-path", default=os.path.join(_DATA, "selected_features_v2"),
                     help="추출된 feature parquet 저장 디렉토리")
_args, _ = _parser.parse_known_args()  # parse_known_args: Jupyter/VSCode 셀 실행 시에도 동작

data_path = _args.data_path
save_path = _args.save_path
os.makedirs(save_path, exist_ok=True)

# %%
_con = duckdb.connect()


def _to_pandas(result):
    """DuckDB 결과 -> Arrow -> pandas (pd.read_parquet(pyarrow)과 같은 변환 경로: 문자열 object, null은 None)."""
    to_arrow = getattr(result, "to_arrow_table", None) or result.fetch_arrow_table  # duckdb >= 1.4 / 이전 버전
    return to_arrow().to_pandas()


def read_parquet_filtered(path, columns, values=None, filter_col="itemid"):
    """
    DuckDB로 parquet에서 필요한 컬럼/행만 읽어 pandas DataFrame으로 반환합니다.

    결과는 `pd.read_parquet(path)[columns]` 후
    `df[df[filter_col].isin(values)]`를 한 것과 같은 행을 같은 순서로 돌려줍니다.
    (index는 0..n-1로 새로 매겨짐. 이후 코드에서 index 값을 쓰지 않으므로 결과에 영향 없음)

    Args:
        path: parquet 파일 경로
        columns: 읽을 컬럼 목록
        values: filter_col 값 목록. None이면 행 필터 없이 컬럼만 선택
        filter_col: 행 필터에 쓸 컬럼 (기본 itemid)
    """
    cols = ", ".join(f'"{c}"' for c in columns)
    src = path.replace("'", "''")
    sql = f"SELECT {cols} FROM read_parquet('{src}', file_row_number=true)"
    params = []
    if values is not None:
        sql += f' WHERE "{filter_col}" IN (SELECT unnest(?::VARCHAR[]))'
        params.append([str(v) for v in values])
    sql += " ORDER BY file_row_number"
    return _to_pandas(_con.execute(sql, params))


def distinct_matching(path, column, predicate):
    """
    column의 고유값 중 predicate(pandas Series -> bool Series)를 만족하는 값의 목록.
    문자열 포함 검색 같은 조건을 고유값에만 pandas로 적용해서 원본과 똑같은 판정을 보장합니다.
    """
    src = path.replace("'", "''")
    uniq = _to_pandas(_con.execute(
        f'SELECT DISTINCT "{column}" AS v FROM read_parquet(\'{src}\')'
    ))["v"]
    return uniq[predicate(uniq)].tolist()
# %%

def get_vital_sign(
    data_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Get vital sign data from MIMIC-IV dataset (admission level).
    """
    
    vital_sign_itemids = {
        # Respiratory rate
        "220210":"resp", 
        
        # Oxygen saturation
        "220277":"spo2", 
        
        # Blood pressure measurements
        # Systolic blood pressure (SBP)
        "220050":"a_sbp",  # Arterial blood pressure
        "220179":"a_sbp",  # Non-invasive blood pressure
        "224167":"a_sbp",  # Left arm
        "227243":"a_sbp",  # Right arm
        
        # Diastolic blood pressure (DBP)
        "220051":"a_dbp",  # Arterial blood pressure
        "220180":"a_dbp",  # Non-invasive blood pressure
        "224643":"a_dbp",  # Left arm
        "227242":"a_dbp",  # Right arm
        
        # Pulse
        "220045":"pulse", 
    }
    
    # Load icustays for hadm_id mapping (chartevents uses stay_id)
    icustays = pd.read_parquet(
        os.path.join(data_path, "icu", "icustays.parquet")
    )[["stay_id", "subject_id", "hadm_id"]]
    
    df_vital = (
        read_parquet_filtered(
            os.path.join(data_path, "icu", "chartevents.parquet"),
            columns=["stay_id", "charttime", "itemid", "valuenum"],
            values=vital_sign_itemids.keys(),
        )
        .merge(icustays, on="stay_id", how="left")
        .drop(columns=["stay_id"])  # Remove stay_id, keep hadm_id
        .assign(
            feature_name = lambda df: df["itemid"].map(vital_sign_itemids),
            valuenum = lambda df: pd.to_numeric(df["valuenum"], errors="coerce"),
            charttime = lambda df: pd.to_datetime(df["charttime"]),
        )
        .pipe(
            lambda df: (
                df.to_parquet(
                    os.path.join(save_path, "vital_sign.parquet"),
                    index=False
                )
                or df
            )
        )
    )
    
    return df_vital


def get_laboratory(
    data_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Get laboratory data from MIMIC-IV dataset (admission level).
    """
    laboratory_itemids = {
        "50912":"creatinine", "52024":"creatinine", "52546":"creatinine",  # Creatinine
        "51301":"wbc", "51755":"wbc", "51756":"wbc", "51300":"wbc",  # White blood cells
        "51652":"crp", "50889":"crp",         # C-reactive protein
        # Procalcitonin - No Procalcitonin itemid in MIMIC-IV
        "51222":"hemoglobin", "51640":"hemoglobin", "51645":"hemoglobin", "50811":"hemoglobin",  # Hemoglobin
        "50810":"hematocrit", "51638":"hematocrit", "51639":"hematocrit", "51221":"hematocrit",  # Hematocrit
        "51265":"platelet", "53189":"platelet",  # Platelets
        "51237":"pt_inr", "51675":"pt_inr",   # Prothrombin time/INR
        "51275":"aptt",                       # Activated partial thromboplastin time
    }
    
    df_laboratory = (
        read_parquet_filtered(
            os.path.join(data_path, "hosp", "labevents.parquet"),
            columns=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
            values=laboratory_itemids.keys(),
        )
        .assign(
            feature_name = lambda df: df["itemid"].map(laboratory_itemids),
            valuenum = lambda df: pd.to_numeric(df["valuenum"], errors="coerce"),
            charttime = lambda df: pd.to_datetime(df["charttime"]),
        )
        .pipe(
            lambda df: (
                df.to_parquet(
                    os.path.join(save_path, "laboratory.parquet"),
                    index=False
                )
                or df
            )
        )
    )
    return df_laboratory

def get_urine_output(
    data_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Get urine output data from MIMIC-IV dataset (admission level).
    """
    urine_output_itemids = {
        "226559":"urine",  # Foley catheter output
        "226560":"urine",   # Void output
    }
    
    # Load icustays for hadm_id mapping (outputevents uses stay_id)
    icustays = pd.read_parquet(
        os.path.join(data_path, "icu", "icustays.parquet")
    )[["stay_id", "subject_id", "hadm_id"]]
    
    df_urine_output = (
        read_parquet_filtered(
            os.path.join(data_path, "icu", "outputevents.parquet"),
            columns=["stay_id", "charttime", "itemid", "value"],
            values=urine_output_itemids.keys(),
        )
        .merge(icustays, on="stay_id", how="left")
        .drop(columns=["stay_id"])  # Remove stay_id, keep hadm_id
        .assign(
            feature_name = lambda df: df["itemid"].map(urine_output_itemids),
            value = lambda df: pd.to_numeric(df["value"], errors="coerce"),
            charttime = lambda df: pd.to_datetime(df["charttime"]),
        )
        .pipe(
            lambda df: (
                df.to_parquet(
                    os.path.join(save_path, "urine_output.parquet"),
                    index=False
                )
                or df
            )
        )
    )
    return df_urine_output


def get_mechanical_ventilation(
    data_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Get mechanical ventilation data from MIMIC-IV dataset (admission level).
    Uses first ICU intime as reference for 24h window.
    """
    
    # Ventilator itemids
    ventilator_itemids = {
        "223849":"vent_mode",
        "220339":"peep",
        "224685":"tidal_volume",
        "224684":"tidal_volume",
        "224686":"tidal_volume",
        "224688":"resp_rate",
        "224690":"resp_rate",
        "223835":"fio2",
    }
    
    df_chartevents = read_parquet_filtered(
        os.path.join(data_path, "icu", "chartevents.parquet"),
        columns=["stay_id", "charttime", "itemid", "valuenum"],
        values=ventilator_itemids.keys(),
    )
    
    # Load icustays for hadm_id mapping and first ICU intime
    df_icustays = pd.read_parquet(
        os.path.join(data_path, "icu", "icustays.parquet")
    )[["stay_id", "subject_id", "hadm_id", "intime"]]
    df_icustays['intime'] = pd.to_datetime(df_icustays['intime'])
    
    # Get first ICU intime per admission
    first_icu_intime = df_icustays.groupby('hadm_id')['intime'].min().reset_index()
    first_icu_intime.columns = ['hadm_id', 'first_icu_intime']
    
    vent_df = df_chartevents[df_chartevents['itemid'].isin(ventilator_itemids.keys())].copy()
    # Merge with icustays for hadm_id
    vent_df = vent_df.merge(df_icustays[['stay_id', 'subject_id', 'hadm_id']], on='stay_id')
    # Merge with first ICU intime
    vent_df = vent_df.merge(first_icu_intime, on='hadm_id')
    
    # 24시간 window 계산 (from first ICU intime)
    vent_df['time_from_icu'] = (
        vent_df['charttime'].astype("datetime64[ns]") - vent_df['first_icu_intime']
    ).dt.total_seconds() / 3600
    
    # 24시간 이내 환기 여부
    vent_24h = vent_df[
        (vent_df['time_from_icu'] >= 0) & 
        (vent_df['time_from_icu'] <= 24)
    ].groupby(['subject_id', 'hadm_id']).size().reset_index(name='vent_events')
    
    vent_24h['mech_vent_24h'] = 1
    
    df_ventilation = (
        vent_24h[['subject_id', 'hadm_id', 'mech_vent_24h']]
        .pipe(
            lambda df: (
                df.to_parquet(
                    os.path.join(save_path, "ventilation.parquet"),
                    index=False
                )
                or df
            )
        )
    )
    
    return df_ventilation

def convert_to_mcgkgmin_vectorized(df):
    """다양한 unit을 mcg/kg/min으로 변환 (vectorized)"""
    
    rate = pd.to_numeric(df['rate'], errors='coerce')
    weight = pd.to_numeric(df['weight_kg'], errors='coerce')
    rateuom = df['rateuom']
    
    # Initialize with NaN
    result = pd.Series(np.nan, index=df.index)
    
    # Valid mask: rate and weight are not NaN and weight > 0
    valid_mask = rate.notna() & weight.notna() & (weight > 0)
    
    # mcg/kg/min - already in correct units
    mask_mcgkgmin = valid_mask & (rateuom == 'mcg/kg/min')
    result.loc[mask_mcgkgmin] = rate.loc[mask_mcgkgmin]
    
    # mcg/min - divide by weight
    mask_mcgmin = valid_mask & (rateuom == 'mcg/min')
    result.loc[mask_mcgmin] = rate.loc[mask_mcgmin] / weight.loc[mask_mcgmin]
    
    # mg/hr - convert to mcg/kg/min
    mask_mghr = valid_mask & (rateuom == 'mg/hr')
    result.loc[mask_mghr] = (rate.loc[mask_mghr] * 1000 / 60) / weight.loc[mask_mghr]
    
    return result


def get_vasopressor(
    data_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Get vasopressor data from MIMIC-IV dataset (admission level).
    """
    vasopressor_itemids = {
        "221906": "norepinephrine", 
        "221289": "epinephrine",
        "221662": "dopamine",
        "222315": "vasopressin",
        "221749": "phenylephrine",
    }
    
    inputevents_df = read_parquet_filtered(
        os.path.join(data_path, "icu", "inputevents.parquet"),
        columns=["stay_id", "starttime", "endtime", "itemid", "rate", "rateuom", "patientweight", "statusdescription"],
        values=vasopressor_itemids.keys(),
    )

    # 아래에서 쓰는 체중 itemid(weight_itemids)만 읽음
    chartevents_df = read_parquet_filtered(
        os.path.join(data_path, "icu", "chartevents.parquet"),
        columns=["stay_id", "itemid", "valuenum"],
        values=["224639", "226512"],
    )
    
    # Load icustays for hadm_id mapping and first ICU intime
    icustays_df = pd.read_parquet(
        os.path.join(data_path, "icu", "icustays.parquet")
    )[["stay_id", "subject_id", "hadm_id", "intime"]]
    icustays_df['intime'] = pd.to_datetime(icustays_df['intime'])
    
    # Get first ICU intime per admission
    first_icu_intime = icustays_df.groupby('hadm_id')['intime'].min().reset_index()
    first_icu_intime.columns = ['hadm_id', 'first_icu_intime']
    
    # Norepinephrine만 필터
    norepi_df = inputevents_df[
        inputevents_df['itemid'].isin([x for x in vasopressor_itemids.keys() if "norepinephrine" in vasopressor_itemids[x]])
    ].copy()    
    
    # Rewritten 제외
    norepi_df = norepi_df[norepi_df['statusdescription'] != 'Rewritten']
    
    # 체중 정보 추출 (by hadm_id)
    weight_itemids = ["224639", "226512"]
    weight_df = chartevents_df[
        chartevents_df['itemid'].isin(weight_itemids)
    ].copy()
    weight_df = weight_df.merge(icustays_df[['stay_id', 'hadm_id']], on='stay_id')
    
    # 체중 평균 계산 (by hadm_id)
    weight_avg = weight_df.astype({'valuenum': 'float64'}).groupby('hadm_id')['valuenum'].mean().reset_index()
    weight_avg.columns = ['hadm_id', 'weight_from_chart']
    
    # Merge with icustays for hadm_id
    norepi_df = norepi_df.merge(icustays_df[['stay_id', 'subject_id', 'hadm_id']], on='stay_id')
    # Merge with first ICU intime
    norepi_df = norepi_df.merge(first_icu_intime, on='hadm_id')
    # Merge weight
    norepi_df = norepi_df.merge(weight_avg, on='hadm_id', how='left')
    
    # 체중 우선순위: patientweight > chartevents
    norepi_df['weight_kg'] = norepi_df['patientweight'].fillna(
        norepi_df['weight_from_chart']
    )
    
    # Dose 계산 - vectorized
    norepi_df['dose_mcgkgmin'] = convert_to_mcgkgmin_vectorized(norepi_df)
    
    # 24시간 window from first ICU intime
    norepi_df['hours_from_icu'] = (
        norepi_df['starttime'].astype("datetime64[ns]") - norepi_df['first_icu_intime']
    ).dt.total_seconds() / 3600
    
    norepi_24h = norepi_df[
        (norepi_df['hours_from_icu'] >= 0) & 
        (norepi_df['hours_from_icu'] <= 24)
    ]
    
    # Features 계산 (admission level)
    features = norepi_24h.astype({"dose_mcgkgmin": "float64", "starttime": "datetime64[ns]", "endtime": "datetime64[ns]"}).groupby(['subject_id', 'hadm_id']).agg({
        'dose_mcgkgmin': ['max', 'mean', 'min'],
        'starttime': 'min',
        'endtime': 'max'
    }).reset_index()
    
    features.columns = [
        'subject_id',
        'hadm_id',
        'norepi_max_dose_24h', 
        'norepi_mean_dose_24h',
        'norepi_min_dose_24h',
        'norepi_start_time',
        'norepi_end_time'
    ]
    
    # Binary flag
    features['norepinephrine_24h'] = 1
    
    df_vasopressor = (
        features[["subject_id", "hadm_id", "norepinephrine_24h"]]
        .pipe(
            lambda df: (
                df.to_parquet(
                    os.path.join(save_path, "vasopressor.parquet"),
                    index=False
                )
                or df
            )
        )
    )
    
    return df_vasopressor


def get_crrt_initiation(
    data_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Get CRRT initiation data from MIMIC-IV dataset (admission level).
    """
    
    # CRRT 했던 stay_id 모으기
    # [교정 5] v1: ["225802", "225803", "224270", "227536", "227525", "226499"]
    crrt_itemids = defs.CRRT_ITEMIDS

    procedureevents = read_parquet_filtered(
        os.path.join(data_path, "icu", "procedureevents.parquet"),
        columns=["stay_id", "itemid"], values=crrt_itemids,
    )
    inputevents = read_parquet_filtered(
        os.path.join(data_path, "icu", "inputevents.parquet"),
        columns=["stay_id", "itemid"], values=crrt_itemids,
    )
    outputevents = read_parquet_filtered(
        os.path.join(data_path, "icu", "outputevents.parquet"),
        columns=["stay_id", "itemid"], values=crrt_itemids,
    )

    # Load icustays for hadm_id mapping
    icustays = pd.read_parquet(
        os.path.join(data_path, "icu", "icustays.parquet")
    )[["stay_id", "subject_id", "hadm_id"]]
    
    crrt_from_proc = set(procedureevents[
        procedureevents['itemid'].isin(crrt_itemids)
    ]['stay_id'])
    
    crrt_from_input = set(inputevents[
        inputevents['itemid'].isin(crrt_itemids)
    ]['stay_id'])
    
    crrt_from_output = set(outputevents[
        outputevents['itemid'].isin(crrt_itemids)
    ]['stay_id'])
    
    all_crrt = crrt_from_proc | crrt_from_input | crrt_from_output
    
    # Convert to hadm_id level
    df_crrt_initiation = (
        pd.DataFrame({'stay_id': list(all_crrt), "crrt_initiation": 1})
        .merge(icustays, on="stay_id", how="left")
        .groupby(['subject_id', 'hadm_id'])['crrt_initiation'].max().reset_index()
        .pipe(
            lambda df: (
                df.to_parquet(
                    os.path.join(save_path, "crrt_initiation.parquet"),
                    index=False
                )
                or df
            )
        )
    )
    
    return df_crrt_initiation


def get_surgery_type(
    data_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Get surgery type data from MIMIC-IV dataset (admission level).
    
    분류:
    - 정규 수술 (Elective surgery): admission_type이 'ELECTIVE'이고 수술 기록이 있는 경우
    - 응급 수술 (Emergency surgery): admission_type이 'EMERGENCY'/'URGENT'이고 수술 기록이 있는 경우
    - 비 수술 (Non-surgical): 수술 기록이 없는 경우
    """
    
    # Load admissions
    admissions = pd.read_parquet(
        os.path.join(data_path, "hosp", "admissions.parquet")
    )[["subject_id", "hadm_id", "admission_type"]]
    
    # [교정 1] v1은 procedureevents itemid 225402/225401/225399도 수술로 봤으나,
    # 실제 라벨은 EKG / Blood Cultured / Lumbar Puncture -> 제거하고 services 기준만 사용
    services = pd.read_parquet(
        os.path.join(data_path, "hosp", "services.parquet")
    )
    all_surgical_hadm = set(
        services[services['curr_service'].isin(defs.SURGICAL_SERVICES)]['hadm_id']
    )
    
    # Work at admission level
    df_surgery = admissions.copy()
    
    # 수술 여부 결정
    df_surgery['is_surgical'] = df_surgery['hadm_id'].isin(all_surgical_hadm)
    
    # 수술 유형 분류 - VECTORIZED
    elective_types = ['ELECTIVE', 'SURGICAL SAME DAY ADMISSION']
    emergency_types = ['EMERGENCY', 'URGENT', 'EW EMER.', 'DIRECT EMER.']
    
    # Default to non_surgical
    df_surgery['surgery_type'] = 'non_surgical'
    
    # Surgical patients with elective admission
    is_elective = df_surgery['is_surgical'] & df_surgery['admission_type'].isin(elective_types)
    df_surgery.loc[is_elective, 'surgery_type'] = 'elective_surgery'
    
    # Surgical patients with emergency admission
    is_emergency = df_surgery['is_surgical'] & df_surgery['admission_type'].isin(emergency_types)
    df_surgery.loc[is_emergency, 'surgery_type'] = 'emergency_surgery'
    
    # Other surgical patients default to elective
    is_other_surgical = df_surgery['is_surgical'] & ~is_elective & ~is_emergency
    df_surgery.loc[is_other_surgical, 'surgery_type'] = 'elective_surgery'
    
    # One-hot encoding (vectorized)
    df_surgery['is_elective_surgery'] = (df_surgery['surgery_type'] == 'elective_surgery').astype(int)
    df_surgery['is_emergency_surgery'] = (df_surgery['surgery_type'] == 'emergency_surgery').astype(int)
    df_surgery['is_non_surgical'] = (df_surgery['surgery_type'] == 'non_surgical').astype(int)
    
    df_surgery_result = (
        df_surgery[["subject_id", "hadm_id", "surgery_type", 
                    "is_elective_surgery", "is_emergency_surgery", "is_non_surgical"]]
        .pipe(
            lambda df: (
                df.to_parquet(
                    os.path.join(save_path, "surgery_type.parquet"),
                    index=False
                )
                or df
            )
        )
    )
    
    return df_surgery_result


def get_transfusion_volume(
    data_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Get transfusion volume data from MIMIC-IV dataset (admission level, vectorized).
    
    수혈량 (mL):
    - RBC (Packed Red Blood Cells): 0-24h, 0-48h, 0-72h 누적량
    - FFP (Fresh Frozen Plasma): 0-24h, 0-48h, 0-72h 누적량  
    - Platelet: 0-24h, 0-48h, 0-72h 누적량
    """
    
    print("Computing transfusion volumes (admission level) - vectorized...")
    
    # Transfusion itemids in MIMIC-IV inputevents
    transfusion_itemids = {
        # RBC (Packed Red Blood Cells)
        "225168": "rbc",  # Packed Red Blood Cells
        "226368": "rbc",  # OR Packed RBC Intake
        "227070": "rbc",  # PACU Packed RBC Intake
        
        # FFP (Fresh Frozen Plasma)
        "220970": "ffp",  # Fresh Frozen Plasma
        "226367": "ffp",  # OR FFP Intake
        "227072": "ffp",  # PACU FFP Intake
        
        # Platelets
        "225170": "platelet",  # Platelets
        "226369": "platelet",  # OR Platelet Intake
        "227071": "platelet",  # PACU Platelet Intake
    }
    
    # Load icustays for hadm_id mapping and first ICU intime
    icustays = pd.read_parquet(
        os.path.join(data_path, "icu", "icustays.parquet")
    )[["stay_id", "subject_id", "hadm_id", "intime"]]
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    
    # Get first ICU intime per admission
    first_icu_intime = icustays.groupby('hadm_id').agg({
        'subject_id': 'first',
        'intime': 'min'
    }).reset_index()
    first_icu_intime.columns = ['hadm_id', 'subject_id', 'first_icu_intime']
    
    # Load inputevents
    inputevents = (
        read_parquet_filtered(
            os.path.join(data_path, "icu", "inputevents.parquet"),
            columns=["stay_id", "starttime", "itemid", "amount", "amountuom"],
            values=transfusion_itemids.keys(),
        )
        .assign(
            amount = lambda df: pd.to_numeric(df["amount"], errors="coerce"),
            starttime = lambda df: pd.to_datetime(df["starttime"]),
        )
    )
    
    # Filter transfusion events
    transfusion_df = inputevents[inputevents['itemid'].isin(transfusion_itemids.keys())].copy()
    transfusion_df['transfusion_type'] = transfusion_df['itemid'].map(transfusion_itemids)
    
    # Merge with icustays for hadm_id
    transfusion_df = transfusion_df.merge(icustays[['stay_id', 'hadm_id']], on='stay_id', how='inner')
    # Merge with first ICU intime
    transfusion_df = transfusion_df.merge(first_icu_intime[['hadm_id', 'first_icu_intime']], on='hadm_id', how='inner')
    
    # Calculate hours from first ICU intime
    transfusion_df['hours_from_icu'] = (
        transfusion_df['starttime'] - transfusion_df['first_icu_intime']
    ).dt.total_seconds() / 3600
    
    # Filter to positive time only (after first ICU admission)
    transfusion_df = transfusion_df[transfusion_df['hours_from_icu'] >= 0].copy()
    
    # Start with first_icu_intime as base (only admissions with ICU stays)
    df_transfusion = first_icu_intime[['subject_id', 'hadm_id']].copy()
    
    # Vectorized calculation - create time window flags
    transfusion_df['within_24h'] = transfusion_df['hours_from_icu'] <= 24
    transfusion_df['within_48h'] = transfusion_df['hours_from_icu'] <= 48
    transfusion_df['within_72h'] = transfusion_df['hours_from_icu'] <= 72
    
    # Calculate all volumes at once using pivot (by hadm_id)
    for transfusion_type in ['rbc', 'ffp', 'platelet']:
        type_data = transfusion_df[transfusion_df['transfusion_type'] == transfusion_type]
        
        if not type_data.empty:
            # 24h
            vol_24h = type_data[type_data['within_24h']].groupby('hadm_id')['amount'].sum()
            df_transfusion = df_transfusion.merge(
                vol_24h.rename(f'{transfusion_type}_volume_24h').reset_index(),
                on='hadm_id', how='left'
            )
            
            # 48h
            vol_48h = type_data[type_data['within_48h']].groupby('hadm_id')['amount'].sum()
            df_transfusion = df_transfusion.merge(
                vol_48h.rename(f'{transfusion_type}_volume_48h').reset_index(),
                on='hadm_id', how='left'
            )
            
            # 72h
            vol_72h = type_data[type_data['within_72h']].groupby('hadm_id')['amount'].sum()
            df_transfusion = df_transfusion.merge(
                vol_72h.rename(f'{transfusion_type}_volume_72h').reset_index(),
                on='hadm_id', how='left'
            )
        else:
            df_transfusion[f'{transfusion_type}_volume_24h'] = 0
            df_transfusion[f'{transfusion_type}_volume_48h'] = 0
            df_transfusion[f'{transfusion_type}_volume_72h'] = 0
    
    # Fill NaN with 0
    volume_cols = [c for c in df_transfusion.columns if '_volume_' in c]
    df_transfusion[volume_cols] = df_transfusion[volume_cols].fillna(0)
    
    df_transfusion_result = (
        df_transfusion
        .pipe(
            lambda df: (
                df.to_parquet(
                    os.path.join(save_path, "transfusion_volume.parquet"),
                    index=False
                )
                or df
            )
        )
    )
    
    print(f"  Completed transfusion volume calculation for {len(df_transfusion_result)} stays")
    
    return df_transfusion_result


def get_mortality(
    data_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Get mortality outcomes from MIMIC-IV dataset (admission level with ICU times).
    
    사망률:
    - ICU mortality: ICU 재실 중 사망 (first_icu_intime ~ last_icu_outtime)
    - Hospital mortality: 병원 퇴원 시 사망
    - 28-day mortality: 첫 ICU 입실 후 28일 이내 사망
    """
    
    # Load icustays for ICU times
    icustays = pd.read_parquet(
        os.path.join(data_path, "icu", "icustays.parquet")
    )[["stay_id", "subject_id", "hadm_id", "intime", "outtime"]]
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    icustays['outtime'] = pd.to_datetime(icustays['outtime'])
    
    # Get first ICU intime and last ICU outtime per admission
    icu_times = icustays.groupby('hadm_id').agg({
        'subject_id': 'first',
        'intime': 'min',      # First ICU admission
        'outtime': 'max',     # Last ICU discharge
    }).reset_index()
    icu_times.columns = ['hadm_id', 'subject_id', 'first_icu_intime', 'last_icu_outtime']
    
    # Load admissions for hospital mortality flag
    admissions = pd.read_parquet(
        os.path.join(data_path, "hosp", "admissions.parquet")
    )[["hadm_id", "hospital_expire_flag", "deathtime"]]
    admissions['deathtime'] = pd.to_datetime(admissions['deathtime'])
    
    # Load patients for date of death
    patients = pd.read_parquet(
        os.path.join(data_path, "hosp", "patients.parquet")
    )[["subject_id", "dod"]]
    patients['dod'] = pd.to_datetime(patients['dod'])
    
    # Merge data
    df_mortality = icu_times.merge(admissions, on='hadm_id', how='left')
    df_mortality = df_mortality.merge(patients, on='subject_id', how='left')
    
    # 1. ICU mortality (ICU 재실 중 사망)
    # Death occurred during ICU stay (between first_icu_intime and last_icu_outtime)
    df_mortality['icu_mortality'] = (
        (df_mortality['deathtime'].notna()) &
        (df_mortality['deathtime'] >= df_mortality['first_icu_intime']) &
        (df_mortality['deathtime'] <= df_mortality['last_icu_outtime'])
    ).astype(int)
    
    # 2. Hospital mortality (병원 내 사망)
    df_mortality['hospital_mortality'] = df_mortality['hospital_expire_flag'].fillna(0).astype(int)
    
    # 3. 28-day mortality (첫 ICU 입실 후 28일 이내 사망)
    df_mortality['days_to_death'] = (
        df_mortality['dod'] - df_mortality['first_icu_intime']
    ).dt.days
    
    df_mortality['mortality_28day'] = (
        (df_mortality['dod'].notna()) &
        (df_mortality['days_to_death'] >= 0) &
        (df_mortality['days_to_death'] <= 28)
    ).astype(int)
    
    df_mortality_result = (
        df_mortality[["subject_id", "hadm_id", "icu_mortality", "hospital_mortality", "mortality_28day"]]
        .pipe(
            lambda df: (
                df.to_parquet(
                    os.path.join(save_path, "mortality.parquet"),
                    index=False
                )
                or df
            )
        )
    )
    
    return df_mortality_result


def get_infection(
    data_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Get infection indicators from MIMIC-IV dataset (admission level, vectorized).
    
    감염 지표:
    - 72시간 이후 blood culture 양성
    - 72시간 이후 항생제 추가 (optional)
    """
    
    print("Computing infection indicators (admission level, from first ICU) - vectorized...")
    
    # Load icustays for first ICU intime
    icustays = pd.read_parquet(
        os.path.join(data_path, "icu", "icustays.parquet")
    )[["stay_id", "subject_id", "hadm_id", "intime"]]
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    
    # Get first ICU intime per admission
    first_icu_intime = icustays.groupby('hadm_id').agg({
        'subject_id': 'first',
        'intime': 'min'
    }).reset_index()
    first_icu_intime.columns = ['hadm_id', 'subject_id', 'first_icu_intime']
    
    # Load microbiologyevents for blood culture
    # 아래 blood culture 조건(spec_type_desc에 'BLOOD' 포함)을 만족하는 검체 종류의 행만 읽음
    micro_path = os.path.join(data_path, "hosp", "microbiologyevents.parquet")
    blood_spec_types = distinct_matching(
        micro_path, "spec_type_desc",
        lambda s: s.str.contains('BLOOD', case=False, na=False),
    )
    microevents = (
        read_parquet_filtered(
            micro_path,
            columns=["subject_id", "hadm_id", "charttime", "spec_type_desc", "org_name"],
            values=blood_spec_types, filter_col="spec_type_desc",
        )
        .assign(
            charttime = lambda df: pd.to_datetime(df["charttime"]),
        )
    )
    
    # Filter for blood cultures
    blood_culture = microevents[
        microevents['spec_type_desc'].str.contains('BLOOD', case=False, na=False)
    ].copy()
    
    # Merge with first ICU intime
    blood_culture = blood_culture.merge(
        first_icu_intime[['hadm_id', 'first_icu_intime']],
        on='hadm_id',
        how='inner'
    )
    
    # Calculate hours from first ICU intime
    blood_culture['hours_from_icu'] = (
        blood_culture['charttime'] - blood_culture['first_icu_intime']
    ).dt.total_seconds() / 3600
    
    # Positive blood culture after 72 hours from first ICU
    positive_after_72h = blood_culture[
        (blood_culture['hours_from_icu'] > 72) &
        (blood_culture['org_name'].notna()) &
        (blood_culture['org_name'] != '')
    ]
    
    positive_hadm = set(positive_after_72h['hadm_id'])
    
    # Load prescriptions for antibiotics
    try:
        # [교정 4] v1: 약명 키워드 부분일치('sulfa', 'azole' 등) -> 비항생제 포함, 주요 항생제 누락
        # v2: 전신 항균제 성분 목록(defs.ANTIBIOTIC_AGENTS)으로 약명을 표준 성분명에 매핑
        rx_path = os.path.join(data_path, "hosp", "prescriptions.parquet")
        uniq_drugs = _to_pandas(_con.execute(
            f"SELECT DISTINCT drug FROM read_parquet('{rx_path.replace(chr(39), chr(39) * 2)}') WHERE drug IS NOT NULL"
        ))["drug"]
        drug_to_agent = {
            d: a for d, a in zip(uniq_drugs, uniq_drugs.map(defs.map_antibiotic_agent)) if a is not None
        }
        prescriptions = (
            read_parquet_filtered(
                rx_path,
                columns=["subject_id", "hadm_id", "starttime", "drug", "drug_type", "route"],
                values=list(drug_to_agent), filter_col="drug",
            )
            .assign(
                starttime = lambda df: pd.to_datetime(df["starttime"]),
            )
        )

        # 비전신 투여 경로(점안/점이/외용/질정/catheter lock/세척 등) 제외
        antibiotics = prescriptions[
            ~prescriptions['route'].isin(defs.NON_SYSTEMIC_ROUTES)
        ].copy()
        antibiotics['agent'] = antibiotics['drug'].map(drug_to_agent)
        
        # Merge with first ICU intime
        antibiotics = antibiotics.merge(
            first_icu_intime[['hadm_id', 'first_icu_intime']],
            on='hadm_id',
            how='inner'
        )
        
        # Calculate hours from first ICU intime
        antibiotics['hours_from_icu'] = (
            antibiotics['starttime'] - antibiotics['first_icu_intime']
        ).dt.total_seconds() / 3600
        
        # [교정 4] "새 항생제" 판정 단위: v1은 약명 문자열(소문자) -> v2는 표준 성분명
        # (예: 'Vancomycin' -> 'Vancomycin Oral Liquid'는 v1에서 새 항생제였으나 v2에서는 같은 성분)
        antibiotics['drug_lower'] = antibiotics['agent']

        # New antibiotic after 72 hours from first ICU - VECTORIZED approach
        antibiotics_before_72h = antibiotics[antibiotics['hours_from_icu'] <= 72]
        antibiotics_after_72h = antibiotics[antibiotics['hours_from_icu'] > 72]
        
        # Get unique drugs per hadm_id before and after 72h
        drugs_before = antibiotics_before_72h.groupby('hadm_id')['drug_lower'].apply(set).reset_index()
        drugs_before.columns = ['hadm_id', 'drugs_before_72h']
        
        drugs_after = antibiotics_after_72h.groupby('hadm_id')['drug_lower'].apply(set).reset_index()
        drugs_after.columns = ['hadm_id', 'drugs_after_72h']
        
        # Merge and find new drugs
        drug_comparison = drugs_after.merge(drugs_before, on='hadm_id', how='left')
        
        # Fill NaN with empty set
        empty_set = set()
        drug_comparison['drugs_before_72h'] = drug_comparison['drugs_before_72h'].fillna('').apply(
            lambda x: x if isinstance(x, set) else empty_set
        )
        
        # Find hadm_id with new drugs after 72h
        has_new_drug = [
            len(after - before) > 0 
            for after, before in zip(
                drug_comparison['drugs_after_72h'], 
                drug_comparison['drugs_before_72h']
            )
        ]
        drug_comparison['has_new_drug'] = has_new_drug
        
        new_antibiotic_hadm = set(drug_comparison[drug_comparison['has_new_drug']]['hadm_id'])
        
        has_antibiotic_data = True
    except Exception as e:
        print(f"Warning: Could not load antibiotic data: {e}")
        new_antibiotic_hadm = set()
        has_antibiotic_data = False
    
    # Create result dataframe (only admissions with ICU stays)
    df_infection = first_icu_intime[['subject_id', 'hadm_id']].copy()
    df_infection['blood_culture_positive_after_72h'] = df_infection['hadm_id'].isin(positive_hadm).astype(int)
    
    if has_antibiotic_data:
        df_infection['new_antibiotic_after_72h'] = df_infection['hadm_id'].isin(new_antibiotic_hadm).astype(int)
    else:
        df_infection['new_antibiotic_after_72h'] = np.nan
    
    df_infection_result = (
        df_infection
        .pipe(
            lambda df: (
                df.to_parquet(
                    os.path.join(save_path, "infection.parquet"),
                    index=False
                )
                or df
            )
        )
    )
    
    print(f"  Completed infection indicator calculation for {len(df_infection_result)} stays")
    
    return df_infection_result


def load_mimic_data(
    data_path: str,
    save_path: str,
):
    """MIMIC-IV에서 필요한 모든 데이터 로드"""
    
    print("Loading MIMIC-IV data...")
    
    # 1. ICU stays
    print("  - ICU stays")
    icustays = (
        pd.read_parquet(
            os.path.join(data_path, "icu", "icustays.parquet")
        )
        [["stay_id", "subject_id", "hadm_id", "intime", "outtime"]]
        .assign(
            intime = lambda df: pd.to_datetime(df['intime']),
            outtime = lambda df: pd.to_datetime(df['outtime']),
        )
    )
    
    # 2. Patients (for age)
    print("  - Patients")
    patients = (
        pd.read_parquet(
            os.path.join(
                data_path,
                "hosp",
                "patients.parquet"
            )
        )
        [["subject_id", "anchor_age"]]
        .assign(
            anchor_age = lambda df: pd.to_numeric(df['anchor_age'], errors='coerce'),
        )
    )
    
    # 3. Chartevents (vital signs, GCS, ventilator settings)
    print("  - Chartevents (this may take a while...)")
    chart_itemids = [
        # PaO2/FiO2
        "50821",  # PaO2
        "223835", "3420", "190",  # FiO2
        
        # Vital signs
        "220052", "220181", "225312",  # MAP
        "223761", "223762",  # Temperature
        "220045",  # Heart rate
        "220210", "224690",  # Respiratory rate
        
        # GCS
        "220739", "223900", "223901",  # GCS Total
        
        # Ventilator
        "720", "223849",  # Ventilator mode
        
        # Weight
        "224639", "226512"  # Weight
    ]
    
    # DuckDB에서 itemid만 미리 거르고, 원래의 query(itemid + valuenum notnull)는 그대로 적용
    chartevents = (
        read_parquet_filtered(
            os.path.join(data_path, "icu", "chartevents.parquet"),
            columns=["stay_id", "charttime", "itemid", "valuenum"],
            values=chart_itemids,
        )
        .assign(
            itemid = lambda df: df['itemid'].astype(str),
        )
        .query(
            "itemid.isin(@chart_itemids) and valuenum.notnull()"
        )
        .assign(
            charttime = lambda df: pd.to_datetime(df['charttime']),
            valuenum = lambda df: pd.to_numeric(df['valuenum'], errors='coerce'),
        )
    )
    
    # 4. Labevents
    print("  - Labevents (this may take a while...)")
    lab_itemids = [
        "50821",  # PaO2
        "51265",  # Platelets
        "50885",  # Bilirubin
        "50912",  # Creatinine
        "50983",  # Sodium
        "50971",  # Potassium
        "51221",  # Hematocrit
        "51301",  # WBC
        "50820",  # pH
        "50818",  # PCO2
        "51006",  # BUN
    ]
    
    labevents = (
        read_parquet_filtered(
            os.path.join(data_path, "hosp", "labevents.parquet"),
            columns=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
            values=lab_itemids,
        )
        .assign(
            itemid = lambda df: df['itemid'].astype(str),
        )
        .query(
            "itemid.isin(@lab_itemids) and valuenum.notnull()"
        )
        .assign(
            charttime = lambda df: pd.to_datetime(df['charttime']),
            valuenum = lambda df: pd.to_numeric(df['valuenum'], errors='coerce'),
        )
    )
    
    # 5. Inputevents (vasopressors)
    print("  - Inputevents")
    vaso_itemids = [
        '221906', '30047', '30120',   # Norepinephrine
        '221289', '30044', '30119',   # Epinephrine
        '221662', '30043', '30307',   # Dopamine
        '222315', '30051', '30128',   # Vasopressin
        '221749', '30127', '30128'    # Phenylephrine
    ]
    # statusdescription != 'Rewritten' 조건은 NULL 처리 차이를 피하려고 pandas query에 그대로 둠
    inputevents = (
        read_parquet_filtered(
            os.path.join(data_path, "icu", "inputevents.parquet"),
            columns=[
                "stay_id",
                "starttime",
                "endtime",
                "itemid",
                "rate",
                "rateuom",
                "amount",
                "amountuom",
                "statusdescription",
            ],
            values=vaso_itemids,
        )
        .assign(
            itemid = lambda df: df['itemid'].astype(str),
        )
        .query(
            "itemid.isin(@vaso_itemids) and statusdescription != 'Rewritten'"
        )
        .assign(
            starttime = lambda df: pd.to_datetime(df['starttime']),
            endtime = lambda df: pd.to_datetime(df['endtime']),
            amount = lambda df: pd.to_numeric(df['amount'], errors='coerce'),
            rate = lambda df: pd.to_numeric(df['rate'], errors='coerce'),
        )
        .drop(columns=["statusdescription"])
    )
    
    # 6. Outputevents (urine output)
    print("  - Outputevents")
    urine_output_itemids = [
        "226559", "226560", "226561", "226584",  # Urine output
        "226563", "226564", "226565", "226567",
        "226557", "226558"
    ]
    outputevents = (
        read_parquet_filtered(
            os.path.join(data_path, "icu", "outputevents.parquet"),
            columns=["stay_id", "charttime", "itemid", "value"],
            values=urine_output_itemids,
        )
        .assign(
            itemid = lambda df: df['itemid'].astype(str),
        )
        .query("itemid.isin(@urine_output_itemids)")
        .assign(
            charttime=lambda df: pd.to_datetime(df['charttime']),
            value=lambda df: pd.to_numeric(df['value'], errors='coerce')
        )
    )
    
    # 7. Admissions (for admission-level processing)
    print("  - Admissions")
    admissions = (
        pd.read_parquet(
            os.path.join(data_path, "hosp", "admissions.parquet")
        )
        [["subject_id", "hadm_id", "admittime", "dischtime"]]
        .assign(
            admittime = lambda df: pd.to_datetime(df['admittime']),
            dischtime = lambda df: pd.to_datetime(df['dischtime']),
        )
    )
    
    # 8. Admission-level ICU times (first intime, last outtime per hadm_id)
    # This merges multiple ICU stays within one admission
    print("  - Computing admission-level ICU times (merging multiple ICU stays)")
    admission_icu_times = icustays.groupby('hadm_id').agg({
        'subject_id': 'first',
        'intime': 'min',      # First ICU admission within the hospitalization
        'outtime': 'max',     # Last ICU discharge within the hospitalization
    }).reset_index()
    admission_icu_times.columns = ['hadm_id', 'subject_id', 'first_icu_intime', 'last_icu_outtime']
    
    # Merge with admissions
    admissions = admissions.merge(
        admission_icu_times[['hadm_id', 'first_icu_intime', 'last_icu_outtime']],
        on='hadm_id',
        how='left'
    )

    # 9. [교정 2] GCS 총점 (측정 시점별 E+V+M)
    print("  - GCS total (E+V+M per charttime)")
    gcs_total = compute_gcs_total(data_path, icustays)

    print("Data loading completed!\n")

    return {
        'icustays': icustays,
        'patients': patients,
        'chartevents': chartevents,
        'labevents': labevents,
        'inputevents': inputevents,
        'outputevents': outputevents,
        'admissions': admissions,
        'admission_icu_times': admission_icu_times,
        'gcs_total': gcs_total,
    }


def compute_gcs_total(data_path, icustays):
    """
    [교정 2] 측정 시점(stay_id, charttime)별 GCS 총점.
    mimic-code (MIT-LCP) concepts/measurement/gcs.sql 과 같은 규칙:
      - 시점별 구성요소 값: eyes(220739), verbal(223900), motor(223901)의 MAX
      - verbal = 'No Response-ETT'(삽관)이면 verbal을 0으로 표시 -> GCS 15로 간주
      - 같은 stay의 직전 측정 시점이 6시간 이내면, 빠진 구성요소를 직전 값으로 보완
      - 그래도 빠진 구성요소는 정상값(eyes 4, verbal 5, motor 6)으로 간주
    v1은 세 구성요소를 하나의 GCS로 보고 최솟값을 취해서, 거의 모든 환자가 GCS <= 4로 계산됐음.

    Returns:
        DataFrame [hadm_id, charttime, gcs]
    """
    raw = read_parquet_filtered(
        os.path.join(data_path, "icu", "chartevents.parquet"),
        columns=["stay_id", "charttime", "itemid", "value", "valuenum"],
        values=[defs.GCS_EYES_ITEMID, defs.GCS_VERBAL_ITEMID, defs.GCS_MOTOR_ITEMID],
    )
    raw["valuenum"] = pd.to_numeric(raw["valuenum"], errors="coerce")
    raw["charttime"] = pd.to_datetime(raw["charttime"])
    is_ett = (raw["itemid"] == defs.GCS_VERBAL_ITEMID) & (raw["value"] == defs.GCS_VERBAL_ETT_VALUE)
    raw.loc[is_ett, "valuenum"] = 0

    comp = (
        raw.pivot_table(index=["stay_id", "charttime"], columns="itemid", values="valuenum", aggfunc="max")
        .rename(columns={
            defs.GCS_EYES_ITEMID: "eyes",
            defs.GCS_VERBAL_ITEMID: "verbal",
            defs.GCS_MOTOR_ITEMID: "motor",
        })
        .reindex(columns=["eyes", "verbal", "motor"])
        .reset_index()
        .sort_values(["stay_id", "charttime"])
        .reset_index(drop=True)
    )

    # 직전 측정 시점(같은 stay, 6시간 이내)
    prev = comp.groupby("stay_id")[["charttime", "eyes", "verbal", "motor"]].shift(1)
    within = prev["charttime"] > comp["charttime"] - pd.Timedelta(hours=defs.GCS_PREV_WINDOW_HOURS)
    prev_e = prev["eyes"].where(within)
    prev_v = prev["verbal"].where(within)
    prev_m = prev["motor"].where(within)

    e, v, m = comp["eyes"], comp["verbal"], comp["motor"]
    total_current_only = m.fillna(6) + v.fillna(5) + e.fillna(4)
    total_with_prev = m.fillna(prev_m).fillna(6) + v.fillna(prev_v).fillna(5) + e.fillna(prev_e).fillna(4)

    comp["gcs"] = np.select(
        [
            v == 0,                          # 현재 삽관 -> 15
            v.isna() & (prev_v == 0),        # 현재 verbal 없음 & 직전 삽관 -> 15
            prev_v == 0,                     # 직전 삽관이었으나 지금은 아님 -> 직전 값 쓰지 않음
        ],
        [15, 15, total_current_only],
        default=total_with_prev,
    )

    gcs_total = comp.merge(icustays[["stay_id", "hadm_id"]], on="stay_id", how="inner")
    return gcs_total[["hadm_id", "charttime", "gcs"]]


def gcs_min_in_window(data, prediction_window):
    """[교정 2] 첫 ICU 입실 후 prediction_window 시간 내 GCS 총점의 최솟값 (hadm_id index)."""
    g = data['gcs_total'].merge(
        data['admissions'][['hadm_id', 'first_icu_intime']], on='hadm_id', how='inner'
    )
    hours = (g['charttime'] - g['first_icu_intime']).dt.total_seconds() / 3600
    g = g[(hours >= 0) & (hours <= prediction_window)]
    return g.groupby('hadm_id')['gcs'].min()


def compute_sofa_all_patients(data, prediction_window=24, n_workers=None):
    """
    Calculate baseline SOFA scores for all patients (admission level, vectorized).
    
    Calculates the FIRST/ADMISSION SOFA score using data from the initial prediction_window
    (default: first 24 hours) after FIRST ICU admission within the hospitalization.
    Multiple ICU stays within one admission are merged.
    
    Args:
        data: Dictionary containing MIMIC-IV dataframes
        prediction_window: Hours from first ICU admission to calculate baseline score (default: 24)
        n_workers: Not used (kept for API compatibility)
    
    Returns:
        DataFrame with baseline SOFA scores for each hospital admission
    """
    
    print("Computing baseline SOFA scores (first 24h from first ICU admission) - vectorized...")
    
    # Item ID mappings
    itemid_map = {
        'pao2': ["50821"],
        'fio2': ["223835", "3420", "190"],
        'platelets': ["51265"],
        'bilirubin': ["50885"],
        'creatinine': ["50912"],
        'gcs': ["220739", "223900", "223901"],
        'map': ["220052", "220181", "225312"],
    }
    
    vasopressor_itemids = {
        'norepinephrine': ["221906", "30047", "30120"],
        'epinephrine': ["221289", "30044", "30119"],
        'dopamine': ["221662", "30043", "30307"],
    }
    
    # 데이터 준비
    chartevents = data['chartevents']
    labevents = data['labevents']
    icustays = data['icustays']
    inputevents = data['inputevents']
    admissions = data['admissions']
    
    # chartevents에 hadm_id 추가 (via icustays)
    print("  Preparing data...")
    chartevents_with_hadm = chartevents.merge(
        icustays[['stay_id', 'hadm_id']],
        on='stay_id',
        how='inner'
    )
    
    # 통합 데이터 (hadm_id 기준)
    all_data = pd.concat([
        chartevents_with_hadm[['hadm_id', 'charttime', 'itemid', 'valuenum']],
        labevents[['hadm_id', 'charttime', 'itemid', 'valuenum']]
    ], ignore_index=True)
    
    # Merge with first ICU intime for time window filtering
    all_data = all_data.merge(
        admissions[['hadm_id', 'first_icu_intime']],
        on='hadm_id',
        how='inner'
    )
    
    # 24시간 윈도우 필터링 (from first ICU intime)
    all_data['hours_from_icu'] = (
        all_data['charttime'] - all_data['first_icu_intime']
    ).dt.total_seconds() / 3600
    
    window_data = all_data[
        (all_data['hours_from_icu'] >= 0) & 
        (all_data['hours_from_icu'] <= prediction_window)
    ].copy()
    
    del all_data  # 메모리 해제
    
    # Ensure valuenum is numeric
    window_data['valuenum'] = pd.to_numeric(window_data['valuenum'], errors='coerce')
    
    print("  Computing SOFA components...")
    
    # 1. Respiration (PaO2/FiO2)
    pao2_itemids = set(itemid_map['pao2'])
    fio2_itemids = set(itemid_map['fio2'])
    
    pao2_data = window_data[window_data['itemid'].isin(pao2_itemids)].groupby('hadm_id')['valuenum'].mean()
    fio2_data = window_data[window_data['itemid'].isin(fio2_itemids)].groupby('hadm_id')['valuenum'].mean()
    
    # FiO2 변환 (>1이면 /100) - vectorized
    fio2_data = fio2_data.where(fio2_data <= 1, fio2_data / 100)
    
    pf_ratio = pao2_data / fio2_data.reindex(pao2_data.index)
    
    sofa_resp = pd.Series(0, index=admissions['hadm_id'])
    sofa_resp.loc[sofa_resp.index.isin(pf_ratio[pf_ratio < 400].index)] = 1
    sofa_resp.loc[sofa_resp.index.isin(pf_ratio[pf_ratio < 300].index)] = 2
    sofa_resp.loc[sofa_resp.index.isin(pf_ratio[pf_ratio < 200].index)] = 3
    sofa_resp.loc[sofa_resp.index.isin(pf_ratio[pf_ratio < 100].index)] = 4
    
    # 2. Coagulation (Platelets - min)
    plt_itemids = set(itemid_map['platelets'])
    plt_min = window_data[window_data['itemid'].isin(plt_itemids)].groupby('hadm_id')['valuenum'].min()
    
    sofa_coag = pd.Series(0, index=admissions['hadm_id'])
    sofa_coag.loc[sofa_coag.index.isin(plt_min[plt_min < 150].index)] = 1
    sofa_coag.loc[sofa_coag.index.isin(plt_min[plt_min < 100].index)] = 2
    sofa_coag.loc[sofa_coag.index.isin(plt_min[plt_min < 50].index)] = 3
    sofa_coag.loc[sofa_coag.index.isin(plt_min[plt_min < 20].index)] = 4
    
    # 3. Liver (Bilirubin - max)
    bili_itemids = set(itemid_map['bilirubin'])
    bili_max = window_data[window_data['itemid'].isin(bili_itemids)].groupby('hadm_id')['valuenum'].max()
    
    sofa_liver = pd.Series(0, index=admissions['hadm_id'])
    sofa_liver.loc[sofa_liver.index.isin(bili_max[bili_max >= 1.2].index)] = 1
    sofa_liver.loc[sofa_liver.index.isin(bili_max[bili_max >= 2.0].index)] = 2
    sofa_liver.loc[sofa_liver.index.isin(bili_max[bili_max >= 6.0].index)] = 3
    sofa_liver.loc[sofa_liver.index.isin(bili_max[bili_max >= 12.0].index)] = 4
    
    # 4. Cardiovascular (MAP + Vasopressors)
    vaso_all_itemids = (
        set(vasopressor_itemids['norepinephrine']) |
        set(vasopressor_itemids['epinephrine']) |
        set(vasopressor_itemids['dopamine'])
    )
    
    # inputevents에서 vasopressor 사용한 hadm_id 추출
    vaso_input = inputevents[inputevents['itemid'].isin(vaso_all_itemids)].merge(
        icustays[['stay_id', 'hadm_id']],
        on='stay_id',
        how='inner'
    ).merge(
        admissions[['hadm_id', 'first_icu_intime']],
        on='hadm_id',
        how='inner'
    )
    vaso_input['hours_from_icu'] = (
        vaso_input['starttime'] - vaso_input['first_icu_intime']
    ).dt.total_seconds() / 3600
    vaso_input_window = vaso_input[
        (vaso_input['hours_from_icu'] >= 0) & 
        (vaso_input['hours_from_icu'] <= prediction_window)
    ]
    
    # Vasopressor type별 hadm_id
    norepi_hadm = set(vaso_input_window[vaso_input_window['itemid'].isin(vasopressor_itemids['norepinephrine'])]['hadm_id'])
    epi_hadm = set(vaso_input_window[vaso_input_window['itemid'].isin(vasopressor_itemids['epinephrine'])]['hadm_id'])
    dopa_hadm = set(vaso_input_window[vaso_input_window['itemid'].isin(vasopressor_itemids['dopamine'])]['hadm_id'])
    
    # MAP
    map_itemids = set(itemid_map['map'])
    map_min = window_data[window_data['itemid'].isin(map_itemids)].groupby('hadm_id')['valuenum'].min()
    
    sofa_cardio = pd.Series(0, index=admissions['hadm_id'])
    low_map_hadm = set(map_min[map_min < 70].index)
    sofa_cardio.loc[sofa_cardio.index.isin(low_map_hadm)] = 1
    sofa_cardio.loc[sofa_cardio.index.isin(dopa_hadm)] = 2
    sofa_cardio.loc[sofa_cardio.index.isin(epi_hadm | norepi_hadm)] = 3
    
    # 5. CNS (GCS - min)
    # [교정 2] v1: E/V/M 구성요소 itemid 값의 최솟값 -> v2: GCS 총점의 최솟값
    gcs_min = gcs_min_in_window(data, prediction_window)
    
    sofa_cns = pd.Series(0, index=admissions['hadm_id'])
    sofa_cns.loc[sofa_cns.index.isin(gcs_min[gcs_min < 15].index)] = 1
    sofa_cns.loc[sofa_cns.index.isin(gcs_min[gcs_min < 13].index)] = 2
    sofa_cns.loc[sofa_cns.index.isin(gcs_min[gcs_min < 10].index)] = 3
    sofa_cns.loc[sofa_cns.index.isin(gcs_min[gcs_min < 6].index)] = 4
    
    # 6. Renal (Creatinine - max)
    cr_itemids = set(itemid_map['creatinine'])
    cr_max = window_data[window_data['itemid'].isin(cr_itemids)].groupby('hadm_id')['valuenum'].max()
    
    sofa_renal = pd.Series(0, index=admissions['hadm_id'])
    sofa_renal.loc[sofa_renal.index.isin(cr_max[cr_max >= 1.2].index)] = 1
    sofa_renal.loc[sofa_renal.index.isin(cr_max[cr_max >= 2.0].index)] = 2
    sofa_renal.loc[sofa_renal.index.isin(cr_max[cr_max >= 3.5].index)] = 3
    sofa_renal.loc[sofa_renal.index.isin(cr_max[cr_max >= 5.0].index)] = 4
    
    # 결과 통합
    result_df = pd.DataFrame({
        'subject_id': admissions['subject_id'].values,
        'hadm_id': admissions['hadm_id'].values,
        'sofa_resp': sofa_resp.values,
        'sofa_coag': sofa_coag.values,
        'sofa_liver': sofa_liver.values,
        'sofa_cardio': sofa_cardio.values,
        'sofa_cns': sofa_cns.values,
        'sofa_renal': sofa_renal.values,
    })
    result_df['sofa_total'] = (
        result_df['sofa_resp'] + result_df['sofa_coag'] + result_df['sofa_liver'] +
        result_df['sofa_cardio'] + result_df['sofa_cns'] + result_df['sofa_renal']
    )
    
    print(f"  Completed SOFA calculation for {len(result_df)} admissions")
    
    return result_df


def _score_temperature_vectorized(temp: pd.Series) -> pd.Series:
    """Vectorized temperature scoring for APACHE II"""
    score = pd.Series(0, index=temp.index)
    score = score.where(~((temp >= 38.5) & (temp < 39)), 1)
    score = score.where(~((temp >= 34) & (temp < 36)), 0)  # normal range
    score = score.where(~((temp >= 32) & (temp < 34)), 2)
    score = score.where(~((temp >= 30) & (temp < 32)), 3)
    score = score.where(~(temp < 30), 4)
    score = score.where(~((temp >= 39) & (temp < 41)), 3)
    score = score.where(~(temp >= 41), 4)
    return score.fillna(0)

def _score_map_vectorized(map_val: pd.Series) -> pd.Series:
    """Vectorized MAP scoring for APACHE II"""
    score = pd.Series(0, index=map_val.index)
    score = score.where(~((map_val >= 110) & (map_val < 130)), 2)
    score = score.where(~((map_val >= 130) & (map_val < 160)), 3)
    score = score.where(~(map_val >= 160), 4)
    score = score.where(~((map_val >= 70) & (map_val < 80)), 0)  # normal
    score = score.where(~((map_val >= 50) & (map_val < 70)), 2)
    score = score.where(~(map_val < 50), 4)
    return score.fillna(0)

def _score_heart_rate_vectorized(hr: pd.Series) -> pd.Series:
    """Vectorized heart rate scoring for APACHE II"""
    score = pd.Series(0, index=hr.index)
    score = score.where(~((hr >= 110) & (hr < 140)), 2)
    score = score.where(~((hr >= 140) & (hr < 180)), 3)
    score = score.where(~(hr >= 180), 4)
    score = score.where(~((hr >= 55) & (hr < 70)), 0)  # low normal
    score = score.where(~((hr >= 40) & (hr < 55)), 2)
    score = score.where(~(hr < 40), 4)
    return score.fillna(0)

def _score_respiratory_rate_vectorized(rr: pd.Series) -> pd.Series:
    """Vectorized respiratory rate scoring for APACHE II"""
    score = pd.Series(0, index=rr.index)
    score = score.where(~((rr >= 25) & (rr < 35)), 1)
    score = score.where(~((rr >= 35) & (rr < 50)), 3)
    score = score.where(~(rr >= 50), 4)
    score = score.where(~((rr >= 10) & (rr < 12)), 0)  # normal
    score = score.where(~((rr >= 6) & (rr < 10)), 2)
    score = score.where(~(rr < 6), 4)
    return score.fillna(0)

def _score_pao2_vectorized(pao2: pd.Series) -> pd.Series:
    """Vectorized PaO2 scoring for APACHE II"""
    score = pd.Series(0, index=pao2.index)
    score = score.where(~((pao2 >= 60) & (pao2 < 70)), 1)
    score = score.where(~((pao2 >= 55) & (pao2 < 60)), 3)
    score = score.where(~(pao2 < 55), 4)
    return score.fillna(0)

def _score_ph_vectorized(ph: pd.Series) -> pd.Series:
    """Vectorized pH scoring for APACHE II"""
    score = pd.Series(0, index=ph.index)
    # High pH
    score = score.where(~((ph >= 7.5) & (ph < 7.6)), 1)
    score = score.where(~((ph >= 7.6) & (ph < 7.7)), 3)
    score = score.where(~(ph >= 7.7), 4)
    # Low pH
    score = score.where(~((ph >= 7.33) & (ph < 7.5)), 0)  # normal
    score = score.where(~((ph >= 7.25) & (ph < 7.33)), 2)
    score = score.where(~((ph >= 7.15) & (ph < 7.25)), 3)
    score = score.where(~(ph < 7.15), 4)
    return score.fillna(0)

def _score_sodium_vectorized(na: pd.Series) -> pd.Series:
    """Vectorized sodium scoring for APACHE II"""
    score = pd.Series(0, index=na.index)
    # High sodium
    score = score.where(~((na >= 150) & (na < 155)), 1)
    score = score.where(~((na >= 155) & (na < 160)), 2)
    score = score.where(~((na >= 160) & (na < 180)), 3)
    score = score.where(~(na >= 180), 4)
    # Low sodium
    score = score.where(~((na >= 130) & (na < 150)), 0)  # normal
    score = score.where(~((na >= 120) & (na < 130)), 2)
    score = score.where(~((na >= 111) & (na < 120)), 3)
    score = score.where(~(na < 111), 4)
    return score.fillna(0)

def _score_potassium_vectorized(k: pd.Series) -> pd.Series:
    """Vectorized potassium scoring for APACHE II"""
    score = pd.Series(0, index=k.index)
    # High potassium
    score = score.where(~((k >= 5.5) & (k < 6)), 1)
    score = score.where(~((k >= 6) & (k < 7)), 3)
    score = score.where(~(k >= 7), 4)
    # Low potassium
    score = score.where(~((k >= 3.5) & (k < 5.5)), 0)  # normal
    score = score.where(~((k >= 3) & (k < 3.5)), 1)
    score = score.where(~((k >= 2.5) & (k < 3)), 2)
    score = score.where(~(k < 2.5), 4)
    return score.fillna(0)

def _score_creatinine_vectorized(cr: pd.Series) -> pd.Series:
    """Vectorized creatinine scoring for APACHE II"""
    score = pd.Series(0, index=cr.index)
    score = score.where(~((cr >= 1.5) & (cr < 2.0)), 2)
    score = score.where(~((cr >= 2.0) & (cr < 3.5)), 3)
    score = score.where(~(cr >= 3.5), 4)
    score = score.where(~(cr < 0.6), 2)
    return score.fillna(0)

def _score_hematocrit_vectorized(hct: pd.Series) -> pd.Series:
    """Vectorized hematocrit scoring for APACHE II"""
    score = pd.Series(0, index=hct.index)
    # High hematocrit
    score = score.where(~((hct >= 46) & (hct < 50)), 1)
    score = score.where(~((hct >= 50) & (hct < 60)), 2)
    score = score.where(~(hct >= 60), 4)
    # Low hematocrit
    score = score.where(~((hct >= 30) & (hct < 46)), 0)  # normal
    score = score.where(~((hct >= 20) & (hct < 30)), 2)
    score = score.where(~(hct < 20), 4)
    return score.fillna(0)

def _score_wbc_vectorized(wbc: pd.Series) -> pd.Series:
    """Vectorized WBC scoring for APACHE II"""
    score = pd.Series(0, index=wbc.index)
    # High WBC
    score = score.where(~((wbc >= 15) & (wbc < 20)), 1)
    score = score.where(~((wbc >= 20) & (wbc < 40)), 2)
    score = score.where(~(wbc >= 40), 4)
    # Low WBC
    score = score.where(~((wbc >= 3) & (wbc < 15)), 0)  # normal
    score = score.where(~((wbc >= 1) & (wbc < 3)), 2)
    score = score.where(~(wbc < 1), 4)
    return score.fillna(0)

def _score_gcs_vectorized(gcs: pd.Series) -> pd.Series:
    """Vectorized GCS scoring for APACHE II (15 - GCS)"""
    return (15 - gcs).clip(lower=0).fillna(0)

def _score_age_vectorized(age: pd.Series) -> pd.Series:
    """Vectorized age scoring for APACHE II"""
    score = pd.Series(0, index=age.index)
    score = score.where(~((age >= 45) & (age < 55)), 2)
    score = score.where(~((age >= 55) & (age < 65)), 3)
    score = score.where(~((age >= 65) & (age < 75)), 5)
    score = score.where(~(age >= 75), 6)
    return score.fillna(0)


def compute_apache2_all_patients(data, prediction_window=24, n_workers=None):
    """
    Calculate baseline APACHE II scores for all patients (admission level, vectorized).
    
    Calculates the FIRST/ADMISSION APACHE II score using worst values from the initial 
    prediction_window (default: first 24 hours) after FIRST ICU admission.
    Multiple ICU stays within one admission are merged.
    
    Args:
        data: Dictionary containing MIMIC-IV dataframes
        prediction_window: Hours from first ICU admission to calculate baseline score (default: 24)
        n_workers: Not used (kept for API compatibility)
    
    Returns:
        DataFrame with baseline APACHE II scores for each hospital admission
    """
    
    print("Computing baseline APACHE II scores (first 24h from first ICU admission) - vectorized...")
    
    # Item ID mappings
    itemid_map = {
        'temperature': ["223761", "223762"],
        'map': ["220052", "220181", "225312"],
        'heart_rate': ["220045"],
        'resp_rate': ["220210", "224690"],
        'pao2': ["50821"],
        'ph': ["50820"],
        'sodium': ["50983"],
        'potassium': ["50971"],
        'creatinine': ["50912"],
        'hematocrit': ["51221"],
        'wbc': ["51301"],
        'gcs': ["220739", "223900", "223901"],
    }
    
    # 데이터 준비
    chartevents = data['chartevents']
    labevents = data['labevents']
    icustays = data['icustays']
    patients = data['patients']
    admissions = data['admissions']
    
    print("  Preparing data...")
    
    # Admissions with age
    admissions_with_age = admissions.merge(patients[['subject_id', 'anchor_age']], on='subject_id')
    
    # chartevents에 hadm_id 추가 (via icustays)
    chartevents_with_hadm = chartevents.merge(
        icustays[['stay_id', 'hadm_id']],
        on='stay_id',
        how='inner'
    )
    
    # 통합 데이터 (hadm_id 기준)
    all_data = pd.concat([
        chartevents_with_hadm[['hadm_id', 'charttime', 'itemid', 'valuenum']],
        labevents[['hadm_id', 'charttime', 'itemid', 'valuenum']]
    ], ignore_index=True)
    
    # Merge with first ICU intime for time window filtering
    all_data = all_data.merge(
        admissions[['hadm_id', 'first_icu_intime']],
        on='hadm_id',
        how='inner'
    )
    
    # 24시간 윈도우 필터링 (from first ICU intime)
    all_data['hours_from_icu'] = (
        all_data['charttime'] - all_data['first_icu_intime']
    ).dt.total_seconds() / 3600
    
    window_data = all_data[
        (all_data['hours_from_icu'] >= 0) & 
        (all_data['hours_from_icu'] <= prediction_window)
    ].copy()
    
    del all_data  # 메모리 해제
    
    # Ensure valuenum is numeric
    window_data['valuenum'] = pd.to_numeric(window_data['valuenum'], errors='coerce')
    
    print("  Computing APACHE II components...")
    
    # 각 측정값의 worst value를 한 번에 계산
    hadm_ids = admissions['hadm_id']
    
    # Helper function to get aggregated values
    def get_agg_values(itemids, agg_func='max'):
        subset = window_data[window_data['itemid'].isin(itemids)]
        if agg_func == 'max':
            return subset.groupby('hadm_id')['valuenum'].max()
        elif agg_func == 'min':
            return subset.groupby('hadm_id')['valuenum'].min()
        else:
            return subset.groupby('hadm_id')['valuenum'].mean()
    
    # 모든 측정값 한 번에 계산
    # [교정 3] v1: °F(223761)와 °C(223762)를 변환 없이 합쳐 max -> v2: °C로 변환 + 유효범위
    temp_rows = window_data[window_data['itemid'].isin(itemid_map['temperature'])]
    t_val = temp_rows['valuenum']
    is_f = (temp_rows['itemid'] == defs.TEMP_F_ITEMID) & t_val.between(*defs.TEMP_F_RANGE, inclusive='neither')
    is_c = (temp_rows['itemid'] == defs.TEMP_C_ITEMID) & t_val.between(*defs.TEMP_C_RANGE, inclusive='neither')
    temp_c = pd.Series(
        np.select([is_f, is_c], [(t_val - 32) / 1.8, t_val], default=np.nan),
        index=temp_rows.index,
    )
    temp_max = temp_c.groupby(temp_rows['hadm_id']).max()
    map_min = get_agg_values(itemid_map['map'], 'min')
    hr_max = get_agg_values(itemid_map['heart_rate'], 'max')
    rr_max = get_agg_values(itemid_map['resp_rate'], 'max')
    pao2_min = get_agg_values(itemid_map['pao2'], 'min')
    ph_min = get_agg_values(itemid_map['ph'], 'min')
    na_max = get_agg_values(itemid_map['sodium'], 'max')
    k_max = get_agg_values(itemid_map['potassium'], 'max')
    cr_max = get_agg_values(itemid_map['creatinine'], 'max')
    hct_max = get_agg_values(itemid_map['hematocrit'], 'max')
    wbc_max = get_agg_values(itemid_map['wbc'], 'max')
    # [교정 2] GCS 총점의 최솟값
    gcs_min = gcs_min_in_window(data, prediction_window)
    
    # 결과 DataFrame 생성
    result_df = pd.DataFrame({'hadm_id': hadm_ids})
    result_df = result_df.set_index('hadm_id')
    
    # 각 컴포넌트 점수 계산
    result_df['temp_score'] = _score_temperature_vectorized(temp_max.reindex(result_df.index))
    result_df['map_score'] = _score_map_vectorized(map_min.reindex(result_df.index))
    result_df['hr_score'] = _score_heart_rate_vectorized(hr_max.reindex(result_df.index))
    result_df['rr_score'] = _score_respiratory_rate_vectorized(rr_max.reindex(result_df.index))
    result_df['pao2_score'] = _score_pao2_vectorized(pao2_min.reindex(result_df.index))
    result_df['ph_score'] = _score_ph_vectorized(ph_min.reindex(result_df.index))
    result_df['na_score'] = _score_sodium_vectorized(na_max.reindex(result_df.index))
    result_df['k_score'] = _score_potassium_vectorized(k_max.reindex(result_df.index))
    result_df['cr_score'] = _score_creatinine_vectorized(cr_max.reindex(result_df.index))
    result_df['hct_score'] = _score_hematocrit_vectorized(hct_max.reindex(result_df.index))
    result_df['wbc_score'] = _score_wbc_vectorized(wbc_max.reindex(result_df.index))
    result_df['gcs_score'] = _score_gcs_vectorized(gcs_min.reindex(result_df.index))
    
    # Age score
    age_series = admissions_with_age.set_index('hadm_id')['anchor_age'].reindex(result_df.index)
    result_df['age_score'] = _score_age_vectorized(age_series)
    
    # Total score
    score_columns = [
        'temp_score', 'map_score', 'hr_score', 'rr_score', 'pao2_score',
        'ph_score', 'na_score', 'k_score', 'cr_score', 'hct_score', 
        'wbc_score', 'gcs_score', 'age_score'
    ]
    result_df['apache2_score'] = result_df[score_columns].sum(axis=1)
    
    # 결과 정리
    result_df = result_df.reset_index()
    # Add subject_id
    result_df = result_df.merge(admissions[['subject_id', 'hadm_id']], on='hadm_id', how='left')
    result_df = result_df[['subject_id', 'hadm_id', 'apache2_score']]
    
    print(f"  Completed APACHE II calculation for {len(result_df)} admissions")
    
    return result_df

def compute_all_severity_scores(data, prediction_window=24, n_workers=None):
    """
    Calculate baseline SOFA + APACHE II scores for all patients (admission level).
    
    Computes FIRST/ADMISSION severity scores using data from the initial prediction_window
    (default: first 24 hours) after hospital admission. Returns one score per admission.
    
    Args:
        data: Dictionary containing MIMIC-IV dataframes
        prediction_window: Hours from admission to calculate baseline scores (default: 24)
        n_workers: Number of parallel workers (default: all CPU cores)
    
    Returns:
        DataFrame with baseline SOFA and APACHE II scores for each hospital admission
    """
    
    print("\n" + "="*60)
    print("COMPUTING BASELINE SEVERITY SCORES (FIRST 24H FROM ADMISSION)")
    print("="*60)
    
    # SOFA
    sofa_scores = compute_sofa_all_patients(data, prediction_window, n_workers)
    
    print()
    
    # APACHE II
    apache_scores = compute_apache2_all_patients(data, prediction_window, n_workers)
    
    # 통합 (admission level)
    severity_scores = data['admissions'][['subject_id', 'hadm_id']].merge(
        sofa_scores, on=['subject_id', 'hadm_id'], how='left'
    ).merge(
        apache_scores, on=['subject_id', 'hadm_id'], how='left'
    )
    
    return severity_scores


def get_sofa_apache(
    data_path: str,
    save_path: str,
    n_workers: int = None,
) -> pd.DataFrame:
    """
    Calculate and save baseline SOFA and APACHE II scores from MIMIC-IV dataset (admission level).
    
    Computes the FIRST/ADMISSION severity scores for each hospital admission using data from
    the first 24 hours after admission. Returns one score per admission.
    
    Args:
        data_path: Path to MIMIC-IV data
        save_path: Path to save processed data
        n_workers: Number of workers for parallel processing (default: all CPU cores)
    
    Returns:
        DataFrame with baseline SOFA and APACHE II scores (one row per admission)
        
    Output columns:
        - stay_id: ICU stay identifier
        - subject_id: Patient identifier
        - hadm_id: Hospital admission identifier
        - sofa_total: Total SOFA score (0-24h)
        - sofa_resp, sofa_coag, sofa_liver, sofa_cardio, sofa_cns, sofa_renal: SOFA components
        - apache2_score: APACHE II score (0-24h)
    """
    
    data = load_mimic_data(data_path=data_path, save_path=save_path)
    
    severity_scores = compute_all_severity_scores(data, prediction_window=24, n_workers=None)
    
    # Save to parquet
    severity_scores.to_parquet(
        os.path.join(save_path, "severity_scores.parquet"),
        index=False
    )
    
    return severity_scores





# %%
if __name__ == "__main__":
    df_vital = get_vital_sign(
        data_path=data_path,
        save_path=save_path
    )
    display(df_vital.head())
    
    df_laboratory = get_laboratory(
        data_path=data_path,
        save_path=save_path
    )
    display(df_laboratory.head())
    
    df_urine_output = get_urine_output(
        data_path=data_path,
        save_path=save_path
    )
    display(df_urine_output.head())
    
    df_ventilation = get_mechanical_ventilation(
        data_path=data_path,
        save_path=save_path
    )
    display(df_ventilation.head())
    
    df_crrt_initiation = get_crrt_initiation(
        data_path=data_path,
        save_path=save_path
    )
    display(df_crrt_initiation.head())
    
    df_sofa_apache = get_sofa_apache(
        data_path=data_path,
        save_path=save_path,
        n_workers=None
    )
    display(df_sofa_apache.head())
    
    df_vasopressor = get_vasopressor(
        data_path=data_path,
        save_path=save_path
    )
    display(df_vasopressor.head())
    
    df_surgery_type = get_surgery_type(
        data_path=data_path,
        save_path=save_path
    )
    display(df_surgery_type.head())
    
    df_transfusion = get_transfusion_volume(
        data_path=data_path,
        save_path=save_path
    )
    display(df_transfusion.head())
    
    df_mortality = get_mortality(
        data_path=data_path,
        save_path=save_path
    )
    display(df_mortality.head())
    
    df_infection = get_infection(
        data_path=data_path,
        save_path=save_path
    )
    display(df_infection.head())
    
    

# %%
