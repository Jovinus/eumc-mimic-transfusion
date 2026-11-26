# v2 (오류 교정판) 변경 내역

- **v1**: 2026-01-17에 만든 기존 데이터셋 (`reference_output/v1_original/`, 코드 `code/02_feature_selection_v1.py`)
- **v2**: `known_issues.md`에서 오류로 확인된 항목을 교정한 데이터셋 (`reference_output/v2_corrected/`, 코드 `code/02_feature_selection.py`, 정의 `code/definitions_v2.py`)
- 대상 입원(85,242건), 컬럼 구성(45개), 나머지 모든 변수는 v1과 **완전히 동일**합니다. 아래 표의 변수만 바뀌었습니다.

## 변경된 변수 요약

| 변수 | v1 | v2 | 값이 바뀐 입원 수 |
|---|---|---|---|
| `surgery_type` 수술 비율 | 61.9% (응급 40.4 / 정규 21.5) | **42.3%** (응급 23.5 / 정규 18.8) | 16,700 |
| `sofa_cns` 평균 ± SD | 4.0 ± 0.3 (99.6%가 4점) | **0.6 ± 1.0** (0점 60.2%, 4점 3.2%) | 82,162 |
| `sofa_total` 평균 ± SD (중앙값) | 7.2 ± 2.7 (7) | **3.8 ± 2.9 (3)** | 82,162 |
| `apache2_score` 평균 ± SD (중앙값) | 27.1 ± 5.7 (27) | **12.4 ± 6.2 (11)** | 85,020 |
| `crrt_initiation` | 4.85% (4,138건) | **2.74%** (2,339건) | 1,799 (모두 1→0) |
| `new_antibiotic_after_72h` | 34.9% | **25.1%** | 11,192 (1→0: 9,764 / 0→1: 1,428) |

## 교정 내용

### [교정 1] 수술 분류 → `surgery_type`, `is_*_surgery`
- **v1 오류**: procedureevents itemid 225402/225401/225399를 수술 기록으로 봤습니다. 실제 라벨은 EKG / Blood Cultured / Lumbar Puncture입니다.
- **v2**: 이 itemid들을 제거하고 `services.curr_service` ∈ {SURG, CSURG, NSURG, ORTHO, TRAUM, TSURG, VSURG, PSURG, GU} 기준만 사용합니다. 서비스 목록과 정규/응급 구분 규칙은 v1과 같습니다.
- 결과: v1에서 수술로 분류됐던 16,700건이 비수술로 바뀌었습니다. 반대 방향 변경은 없습니다.

### [교정 2] GCS → `sofa_cns`, `sofa_total`, `apache2_score`
- **v1 오류**: GCS 구성요소(eye 1–4 / verbal 1–5 / motor 1–6) itemid를 하나의 GCS로 보고 최솟값을 취했습니다. 그래서 거의 모든 환자가 GCS ≤ 4가 되었습니다.
- **v2**: 측정 시점별로 총점(E+V+M)을 만든 뒤 첫 ICU 입실 후 24시간 내 최솟값을 사용합니다. 규칙은 MIMIC 공식 코드 저장소(MIT-LCP mimic-code)의 `concepts/measurement/gcs.sql`을 따릅니다.
  - verbal = `No Response-ETT`(삽관 상태)이면 **GCS 15로 간주**합니다. 진정/삽관 환자를 정상으로 보는 SOFA 관례입니다.
  - 빠진 구성요소는 같은 ICU stay에서 6시간 이내 직전 측정값으로 보완합니다. 그래도 없으면 정상값(E4/V5/M6)으로 간주합니다.
  - 직전 측정이 삽관 상태였고 현재는 아니면, 직전 값을 쓰지 않습니다.
- SOFA CNS 기준(15: 0점, 13–14: 1, 10–12: 2, 6–9: 3, <6: 4)과 APACHE II GCS 점수(15 − GCS)는 v1과 같습니다.
- ⚠ **검토 필요**: "삽관 = GCS 15" 관례 때문에, 삽관 상태에서 motor 반응도 없는 깊은 혼수 환자도 15로 계산됩니다. 대안으로는 삽관 시 verbal을 1로 두거나 E+M만으로 verbal을 추정하는 방법이 있습니다. 연구 목적에 맞게 정해 주세요. 바꾸려면 `02_feature_selection.py`의 `compute_gcs_total()`만 수정하면 됩니다.

### [교정 3] APACHE II 체온 → `apache2_score`
- **v1 오류**: 223761(°F)과 223762(°C)를 변환 없이 합쳐 최댓값을 취했습니다. °F 기록이 있으면 대부분 4점을 받았습니다.
- **v2**: mimic-code `concepts/measurement/vitalsign.sql`과 같은 방식을 씁니다. °F는 70–120 범위만 (°F−32)/1.8로 변환하고, °C는 10–50 범위만 사용합니다. 범위 밖은 오기입으로 보고 제외합니다(°C 항목에 98.5 같은 °F 값이 들어간 경우 등).
- 24시간 최고 체온만 보는 방식(저체온 미반영)은 v1과 같습니다.
- APACHE II 변화는 교정 2(GCS)와 교정 3(체온)이 합쳐진 결과입니다. 항목별 기여도는 따로 계산하지 않았습니다.

### [교정 4] 항생제 → `new_antibiotic_after_72h`
- **v1 오류**:
  - 약명 키워드 부분일치 때문에 비항생제가 포함됐습니다. `'sulfa'` → magnesium/morphine/ferrous sulfate, `'azole'` → pantoprazole/omeprazole(PPI), 항진균제가 걸렸습니다.
  - 주요 항생제가 빠졌습니다. **cefazolin**(69,095건), ceftazidime, cephalexin, cefpodoxime, aztreonam, ertapenem 등입니다.
  - 안약/외용제/질정/catheter lock 같은 비전신 제형이 포함됐습니다.
  - "새 항생제"를 약명 문자열로 비교해서 `Vancomycin` → `Vancomycin Oral Liquid`처럼 제형만 바뀌어도 새 항생제가 됐습니다.
- **v2**:
  1. 전신 항균제 73개 성분 목록(`definitions_v2.ANTIBIOTIC_AGENTS`)으로 약명을 **표준 성분명**에 매핑합니다.
  2. 약명(ophth/otic/cream/oint/gel/vag/lock/농도% 표기 등)이나 route(점안/점이/외용/질/lock/세척/안구 내 주사)가 비전신이면 제외합니다.
  3. "새 항생제" = 첫 ICU 입실 72시간 이후 처방된 성분 중, 72시간 이전(포함)에 없던 성분입니다.
- 제외한 약: 항진균제, 항바이러스제, 항기생충/PJP 예방약(dapsone, atovaquone, pentamidine 등), 비흡수성 장관 항생제(rifaximin, neomycin, paromomycin), 외용 bacitracin, methenamine, demeclocycline(SIADH 치료 목적)
- 결과: 9,764건이 1→0(비항생제나 제형 변경이 원인), 1,428건이 0→1(cefazolin 등 누락됐던 항생제가 원인)

### [교정 5] CRRT → `crrt_initiation`
- **v1 오류**: 224270 **Dialysis Catheter**(카테터 삽입 처치, 4,464 stay)를 CRRT로 봤습니다. 226499(Hemodialysis Output)는 chartevents 항목이라 실제로는 매칭되지 않았습니다.
- **v2**: 두 itemid를 제거했습니다. 남은 정의는 225802 Dialysis-CRRT, 225803 CVVHD, 227536 KCl(CRRT), 227525 Calcium Gluconate(CRRT)입니다.
- 결과: 1,799건 1→0 (카테터만 있고 CRRT 기록은 없는 입원)

## v2에서도 바꾸지 않은 항목 (연구 설계상 결정 필요)
`known_issues.md`의 "정의 확인이 필요한 항목"은 v1 정의를 그대로 유지했습니다.
- `crrt_initiation`의 시간 조건 (현재는 입원 중 언제든)
- `mech_vent_24h`: FiO2/PEEP 기록만으로도 1 (비침습 환기 포함 가능성)
- `*_first` lab: 입원 중 첫 값 (ICU 입실 이후 값이 아님)
- SOFA 세부 규칙(PaO2/FiO2 24h 평균, 승압제 용량 미적용, urine output 미적용), APACHE II 단방향 극값
- 코호트 제외 기준 없음, 나이/성별/체중 변수 없음
