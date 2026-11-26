"""
Step 1. MIMIC-IV v3.1 원본(.csv.gz) -> parquet 변환 (DuckDB)

PhysioNet에서 받은 MIMIC-IV v3.1 원본(.csv.gz)을 parquet로 변환합니다.
모든 컬럼을 문자열(VARCHAR)로 읽어 저장합니다. 이후 단계의 코드는 이 형식을 전제로
작성되어 있습니다(예: itemid를 문자열로 비교). 그러니 다른 방식으로 변환한 parquet는
사용하지 마세요.

Usage:
    python 01_convert_to_parquet.py \
        --src  <mimic-iv-3.1 디렉토리 (hosp/, icu/ 하위에 *.csv.gz)> \
        --dst  <parquet 저장 디렉토리>

    # 전체 테이블을 변환하려면 --all

Original: code/aitrics/clean/convert_extension.py
(pandas 원본: code/legacy_pandas/01_convert_to_parquet.py)

[DuckDB 고속화]
pandas `read_csv(dtype=str)`와 같은 결과가 나오도록 설정했습니다.
  - 모든 컬럼 문자열(all_varchar)
  - 빈 값과 pandas 기본 NA 문자열('NA', 'NULL', 'nan', 'N/A' 등)은 NULL로 처리
  - 원본 행 순서 유지 (preserve_insertion_order)
"""
import argparse
import os
import time
from pathlib import Path

import duckdb
from pandas._libs.parsers import STR_NA_VALUES  # pandas read_csv 기본 NA 문자열 목록

# Transfusion 파이프라인(Step 2, 3)에서 실제로 읽는 테이블
REQUIRED_TABLES = [
    "hosp/admissions",
    "hosp/patients",
    "hosp/labevents",
    "hosp/microbiologyevents",
    "hosp/prescriptions",
    "hosp/services",
    "icu/icustays",
    "icu/chartevents",
    "icu/inputevents",
    "icu/outputevents",
    "icu/procedureevents",
]


def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def convert_extension(src: str, dst: str, convert_all: bool = False) -> None:
    file_path_list = sorted(Path(src).rglob("*.csv.gz"))

    if not convert_all:
        wanted = set(REQUIRED_TABLES)
        file_path_list = [
            p for p in file_path_list
            if f"{p.parent.name}/{p.name.replace('.csv.gz', '')}" in wanted
        ]
        found = {f"{p.parent.name}/{p.name.replace('.csv.gz', '')}" for p in file_path_list}
        missing = wanted - found
        if missing:
            raise FileNotFoundError(f"원본 디렉토리에 다음 테이블이 없습니다: {sorted(missing)}")

    na_list = "[" + ", ".join(_sql_str(v) for v in sorted(STR_NA_VALUES | {""})) + "]"

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = true")

    for file_path in file_path_list:
        sub_dir = file_path.parent.name                       # 'hosp' or 'icu'
        file_name = file_path.name.replace(".csv.gz", "")
        out_dir = os.path.join(dst, sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{file_name}.parquet")

        t0 = time.time()
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv(
                    {_sql_str(str(file_path))},
                    header = true,
                    delim = ',',
                    quote = '"',
                    escape = '"',
                    all_varchar = true,
                    nullstr = {na_list},
                    allow_quoted_nulls = true,
                    max_line_size = 100000000
                )
            ) TO {_sql_str(out_path)} (FORMAT parquet, COMPRESSION zstd)
        """)
        print(f"  {sub_dir}/{file_name}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    # 기본 경로: 이 스크립트 위치 기준 ../data
    _DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=os.path.join(_DATA, "mimic-iv-3.1"), help="MIMIC-IV v3.1 원본 디렉토리")
    parser.add_argument("--dst", default=os.path.join(_DATA, "mimic-iv-parquet"), help="parquet 저장 디렉토리")
    parser.add_argument("--all", action="store_true", help="필요 테이블만이 아니라 전체 테이블 변환")
    args = parser.parse_args()

    convert_extension(args.src, args.dst, convert_all=args.all)
