# analysis_data.parquet 변수 정의 (v2 기준)

> 이 문서는 **v2(오류 교정판)** 기준입니다. v1과 다른 변수는 🔁로 표시했고, v1 정의를 함께 적었습니다. 변경 내역 전체는 [`v2_changes.md`](v2_changes.md)를 참고하세요.

- **단위(row)**: ICU 입실이 1회 이상 있는 입원(`hadm_id`) 1건 = 1 row
- **기준 시점(t0)**: 해당 입원에서 첫 ICU 입실 시각, 즉 `min(icustays.intime)`. 한 입원 안에 ICU stay가 여러 번이면 하나로 합쳐서 봅니다.
- **규모**: 85,242 admissions / 65,366 patients / 45 columns
- **코호트 제외 기준 없음**: 연령, 재입원, ICU 재실기간 등으로 제외하지 않았습니다. 같은 환자의 여러 입원이 모두 포함됩니다.
- itemid 라벨은 MIMIC-IV v3.1 `d_items`, `d_labitems` 기준입니다.

## ID
| 변수 | 정의 |
|---|---|
| `subject_id`, `hadm_id` | MIMIC-IV 식별자 (문자열) |

## Vital signs: t0 ~ t0+24h 평균 (`chartevents`)
| 변수 | itemid |
|---|---|
| `a_sbp_24h_mean` | 220050 Arterial BP systolic, 220179 NIBP systolic, 224167/227243 Manual BP systolic L/R (침습·비침습 혼합) |
| `a_dbp_24h_mean` | 220051, 220180, 224643, 227242 (위와 대응되는 diastolic) |
| `pulse_24h_mean` | 220045 Heart Rate |
| `resp_24h_mean` | 220210 Respiratory Rate |
| `spo2_24h_mean` | 220277 O2 saturation pulseoxymetry |

## Laboratory: 입원 중 **첫 측정값** (`labevents`)
> 주의: 이름은 "first"이지만 ICU 기간이 아니라 **입원 전체에서 가장 먼저 측정된 값**입니다. ICU 입실 전(응급실, 병동) 값일 수 있습니다.

| 변수 | itemid |
|---|---|
| `creatinine_first` | 50912, 52024 (whole blood), 52546 |
| `wbc_first` | 51300, 51301, 51755, 51756 |
| `crp_first` | 50889 CRP, 51652 hs-CRP (결측 91.6%) |
| `hemoglobin_first` | 51222, 51640, 50811 (blood gas), 51645 (calculated) |
| `hematocrit_first` | 51221, 51638, 51639, 50810 (calculated) |
| `platelet_first` | 51265, 53189 |
| `pt_inr_first` | 51237, 51675 |
| `aptt_first` | 51275 PTT |

## Urine output
| 변수 | 정의 |
|---|---|
| `urine_output_24h_cumulative` | t0~t0+24h `outputevents` 226559 Foley + 226560 Void 합계 (mL). 기록이 없으면 NaN |

## 수술 유형 (입원 단위)
| 변수 | 정의 |
|---|---|
| `surgery_type` | `elective_surgery` / `emergency_surgery` / `non_surgical` |
| `is_elective_surgery`, `is_emergency_surgery`, `is_non_surgical` | 위 값의 one-hot 인코딩 |

- 🔁 "수술 환자" = `services.curr_service` ∈ {SURG, CSURG, NSURG, ORTHO, TRAUM, TSURG, VSURG, PSURG, GU}
  - v1: 여기에 더해 `procedureevents` itemid ∈ {225402, 225401, 225399}(실제로는 EKG / Blood Cultured / Lumbar Puncture)도 수술로 봄
- 수술 환자 중 `admission_type` ∈ {EMERGENCY, URGENT, EW EMER., DIRECT EMER.}이면 emergency, 나머지는 elective

## 중증도 점수: t0~t0+24h
| 변수 | 정의 |
|---|---|
| `sofa_resp` | 평균 PaO2(lab 50821) / 평균 FiO2(chart 223835, >1이면 /100). ventilation 여부는 고려하지 않음 |
| `sofa_coag` | 최저 platelet (51265) |
| `sofa_liver` | 최고 bilirubin (50885) |
| `sofa_cardio` | 최저 MAP < 70 → 1점, dopamine 사용 → 2점, norepinephrine 또는 epinephrine 사용 → 3점 (용량 기준은 미적용, 4점 없음) |
| `sofa_cns` 🔁 | 24h 내 **GCS 총점**(측정 시점별 E+V+M, mimic-code `gcs.sql` 규칙: 삽관 시 15, 빠진 항목은 6h 내 직전 값/정상값으로 보완)의 최솟값. v1: E/V/M 구성요소 값의 최솟값 |
| `sofa_renal` | 최고 creatinine (50912). urine output 기준은 미적용 |
| `sofa_total` 🔁 | 위 6개 점수의 합 (sofa_cns 교정 반영) |
| `apache2_score` | APS(temp, MAP, HR, RR, PaO2, pH, Na, K, Cr, Hct, WBC, GCS) + age point. 각 항목은 한 방향의 극값만 사용(temp/HR/RR/Na/K/Cr/Hct/WBC는 max, MAP/PaO2/pH/GCS는 min). A-a gradient, Chronic health point, ARF 가중치는 미적용. 결측 항목은 0점. 나이는 `anchor_age`. 🔁 v2: GCS는 위 GCS 총점 사용, 체온은 °F→°C 변환 후 유효범위(°F 70–120, °C 10–50)만 사용. v1: GCS 구성요소 최솟값, °F/°C 혼재 |

## 처치 (binary, 해당 기록이 없으면 0)
| 변수 | 정의 |
|---|---|
| `mech_vent_24h` | t0~t0+24h에 ventilator 관련 chartevents(223849 Ventilator Mode, 220339 PEEP set, 224684/224685/224686 Tidal Volume, 224688/224690 RR set/total, 223835 FiO2)가 하나라도 있으면 1 ⚠ `known_issues.md` #6 |
| `norepinephrine_24h` | t0~t0+24h에 시작된 norepinephrine(221906) 투여 기록(Rewritten 제외)이 있으면 1 |
| `crrt_initiation` 🔁 | **입원 중 언제든** 225802 Dialysis-CRRT, 225803 CVVHD, 227536 KCl(CRRT), 227525 Ca gluconate(CRRT) 기록이 있으면 1. v1: 224270 Dialysis Catheter(카테터 삽입), 226499도 포함 ⚠ 시간 정의는 `known_issues.md` #5-1 |

## 수혈량 (mL, t0 이후 누적; 수혈하지 않았으면 0)
| 변수 | itemid (`inputevents.amount`, `starttime` 기준) |
|---|---|
| `rbc_volume_{24,48,72}h` | 225168 Packed RBC, 226368 OR Packed RBC, 227070 PACU Packed RBC |
| `ffp_volume_{24,48,72}h` | 220970 FFP, 226367 OR FFP, 227072 PACU FFP |
| `platelet_volume_{24,48,72}h` | 225170 Platelets, 226369 OR Platelet, 227071 PACU Platelet |

- 구간: 0 ≤ (starttime − t0) ≤ 24/48/72h, 누적량
- ICU 밖(OR/PACU) 기록이라도 inputevents에 있고 t0 이후면 포함됩니다.

## 결과 변수
| 변수 | 정의 |
|---|---|
| `icu_mortality` | `admissions.deathtime`이 [첫 ICU intime, 마지막 ICU outtime] 안에 있음 |
| `hospital_mortality` | `admissions.hospital_expire_flag` |
| `mortality_28day` | `patients.dod` − t0 이 0~28일 (`.dt.days` 기준이라 28일 23시간까지 포함). dod에는 퇴원 후 사망도 포함 |
| `blood_culture_positive_after_72h` | t0+72h 이후 `spec_type_desc`에 'BLOOD'가 포함된 검체에서 `org_name`이 기록됨 |
| `new_antibiotic_after_72h` 🔁 | t0+72h 이후 처방된 **전신 항균제 성분** 중 72h 이전(포함)에 없던 성분이 있으면 1. 성분 목록과 비전신 제외 규칙은 `code/definitions_v2.py`. v1: 키워드 부분일치 약명(비항생제 포함, cefazolin 등 누락), 약명 문자열 기준 비교 |

## 정제 (Step 3 마지막 단계)
아래 범위를 벗어난 값은 NaN으로 바꿉니다: SBP 20–300, DBP 10–200, HR 20–300, RR 1–60, SpO2 0–100, Cr 0–30, WBC 0–500, CRP 0–500, Hb 1–25, Hct 1–70, PLT 0–2000, INR 0.1–20, aPTT 0–200, UO 0–20,000 mL, 수혈 24/48/72h 각각 0–50,000/100,000/150,000 mL.
