"""
v2 교정 정의 모음 (02_feature_selection.py에서 import)

v1(원본) 대비 교정 내용과 근거는 docs/v2_changes.md에 정리되어 있습니다.
"""
import re

# ---------------------------------------------------------------------------
# [교정 1] 수술 분류
# v1: services 기준 + procedureevents itemid 225402/225401/225399
#     (주석에는 Surgical Procedure/OR Admission/General Surgery로 되어 있었지만
#      실제로는 EKG/Blood Cultured/Lumbar Puncture)
# v2: services 기준만 사용 (서비스 목록은 v1과 동일)
# ---------------------------------------------------------------------------
SURGICAL_SERVICES = ['SURG', 'CSURG', 'NSURG', 'ORTHO', 'TRAUM', 'TSURG', 'VSURG', 'PSURG', 'GU']

# ---------------------------------------------------------------------------
# [교정 5] CRRT
# v1: 225802, 225803, 224270, 227536, 227525, 226499
# v2: 224270 (Dialysis Catheter: 카테터 삽입 처치) 제거
#     226499 (Hemodialysis Output: chartevents 항목이라 실제로 매칭되지 않음, 정의상 제거)
# ---------------------------------------------------------------------------
CRRT_ITEMIDS = [
    "225802",  # Dialysis - CRRT (procedureevents)
    "225803",  # Dialysis - CVVHD (procedureevents)
    "227536",  # KCl (CRRT) (inputevents)
    "227525",  # Calcium Gluconate (CRRT) (inputevents)
]

# ---------------------------------------------------------------------------
# [교정 3] 체온 (APACHE II)
# mimic-code (MIT-LCP) concepts/measurement/vitalsign.sql 의 유효범위/변환 방식
# ---------------------------------------------------------------------------
TEMP_F_ITEMID = "223761"   # Temperature Fahrenheit: 70 < v < 120 -> (v - 32) / 1.8
TEMP_C_ITEMID = "223762"   # Temperature Celsius:    10 < v < 50
TEMP_F_RANGE = (70, 120)
TEMP_C_RANGE = (10, 50)

# ---------------------------------------------------------------------------
# [교정 2] GCS
# mimic-code (MIT-LCP) concepts/measurement/gcs.sql 방식
# ---------------------------------------------------------------------------
GCS_EYES_ITEMID = "220739"
GCS_VERBAL_ITEMID = "223900"
GCS_MOTOR_ITEMID = "223901"
GCS_VERBAL_ETT_VALUE = "No Response-ETT"   # 삽관 상태 -> GCS 15로 간주 (sedation/intubation 관례)
GCS_PREV_WINDOW_HOURS = 6                  # 빠진 구성요소는 6시간 이내 직전 측정값으로 보완

# ---------------------------------------------------------------------------
# [교정 4] 항생제 (new_antibiotic_after_72h)
# v1: 약명 키워드('sulfa', 'azole', 'mycin' 등) 부분일치
#     -> magnesium/morphine sulfate, PPI(pantoprazole 등), 항진균제가 포함되고,
#        cefazolin/ceftazidime/cephalexin/aztreonam/ertapenem 등은 빠짐
#     -> "새 항생제"를 약명 문자열로 비교해서 같은 약의 제형만 바뀌어도 새 항생제로 판정
# v2: (a) 전신 항균제(antibacterial) 성분 목록으로 판정 (첫 번째로 매칭되는 패턴의 성분명으로 표준화)
#     (b) 비전신 투여(안약/귀약/외용/질정/catheter lock)는 약명 또는 route로 제외
#     (c) "새 항생제" = 72h 이후 처방 성분 중 72h 이전(포함)에 없던 성분
# 제외: 항진균제, 항바이러스제, 항기생충/PJP 예방약(dapsone, atovaquone, pentamidine 등),
#       비흡수성 장관 항생제(rifaximin, neomycin, paromomycin), 외용제(bacitracin 등),
#       methenamine(요로 소독제), demeclocycline(SIADH 치료 목적)
# 순서 중요: 복합제/긴 이름을 먼저 둠 (예: dicloxacillin -> oxacillin, levofloxacin -> ofloxacin)
# ---------------------------------------------------------------------------
ANTIBIOTIC_AGENTS = [
    # Penicillins
    (r"piperacillin", "piperacillin-tazobactam"),
    (r"ampicillin-sulbact|ampicillin/sulbact", "ampicillin-sulbactam"),
    (r"ampicillin", "ampicillin"),
    (r"amoxicillin-clav|amoxicillin-pot clav|amoxicillin/clav", "amoxicillin-clavulanate"),
    (r"amoxic", "amoxicillin"),
    (r"nafcillin", "nafcillin"),
    (r"dicloxacillin", "dicloxacillin"),
    (r"oxacillin", "oxacillin"),
    (r"penicillin", "penicillin"),
    # Cephalosporins
    (r"cefazolin", "cefazolin"),
    (r"cephalexin", "cephalexin"),
    (r"cefadroxil", "cefadroxil"),
    (r"cefoxitin", "cefoxitin"),
    (r"cefuroxime", "cefuroxime"),
    (r"cefpodo", "cefpodoxime"),
    (r"cefdinir", "cefdinir"),
    (r"cefixime", "cefixime"),
    (r"cefotaxime", "cefotaxime"),
    (r"ceftriaxone", "ceftriaxone"),
    (r"ceftazidime-avibactam|avycaz", "ceftazidime-avibactam"),
    (r"ceftazidime", "ceftazidime"),
    (r"cefepime", "cefepime"),
    (r"ceftaroline", "ceftaroline"),
    (r"ceftolozane", "ceftolozane-tazobactam"),
    (r"cefiderocol", "cefiderocol"),
    # Carbapenems / monobactam
    (r"meropenem", "meropenem"),
    (r"imipenem", "imipenem"),
    (r"ertapenem", "ertapenem"),
    (r"doripenem", "doripenem"),
    (r"aztreonam", "aztreonam"),
    # Glycopeptides / lipopeptides / oxazolidinones
    (r"vancomycin|vancocin", "vancomycin"),
    (r"daptomycin", "daptomycin"),
    (r"dalbavancin", "dalbavancin"),
    (r"oritavancin", "oritavancin"),
    (r"telavancin", "telavancin"),
    (r"linezolid", "linezolid"),
    (r"tedizolid", "tedizolid"),
    # Aminoglycosides
    (r"gentamicin", "gentamicin"),
    (r"tobramycin", "tobramycin"),
    (r"amikacin", "amikacin"),
    (r"streptomycin", "streptomycin"),
    # Fluoroquinolones
    (r"ciprofloxacin", "ciprofloxacin"),
    (r"levofloxacin", "levofloxacin"),
    (r"moxifloxacin", "moxifloxacin"),
    (r"ofloxacin", "ofloxacin"),
    (r"gatifloxacin", "gatifloxacin"),
    # Macrolides
    (r"azithromycin", "azithromycin"),
    (r"clarithromycin", "clarithromycin"),
    (r"erythromycin", "erythromycin"),
    (r"fidaxomicin", "fidaxomicin"),
    # Tetracyclines
    (r"doxycycline", "doxycycline"),
    (r"minocycline", "minocycline"),
    (r"tigecycline", "tigecycline"),
    (r"omadacycline", "omadacycline"),
    (r"^tetracycline", "tetracycline"),
    # Others
    (r"clindamycin", "clindamycin"),
    (r"metronidazole", "metronidazole"),
    (r"tinidazole", "tinidazole"),
    (r"sulfameth|^sulfame|trimethoprim-sulfa", "sulfamethoxazole-trimethoprim"),
    (r"^trimethoprim$", "trimethoprim"),
    (r"sulfadiazine", "sulfadiazine"),
    (r"nitrofur", "nitrofurantoin"),
    (r"fosfomycin", "fosfomycin"),
    (r"colistin", "colistin"),
    (r"^polymyxin b sulfate$", "polymyxin b"),
    (r"chloramphenicol", "chloramphenicol"),
    (r"quinupristin", "quinupristin-dalfopristin"),
    (r"rifampin|^rifamp", "rifampin"),
    (r"rifabutin", "rifabutin"),
    (r"rifapentine", "rifapentine"),
    (r"isoniazid", "isoniazid"),
    (r"ethambutol", "ethambutol"),
    (r"pyrazinamide", "pyrazinamide"),
]

# 약명으로 판단하는 비전신 제형 (안약/귀약/외용/질정/catheter lock 등)
NON_SYSTEMIC_NAME_PATTERN = (
    r"ophth|opht|opth|otic|ear drop|cream|oint|\bgel\b|topical|tpoical|vag|shampoo|powder"
    r"|\block\b|silver sulfadiazine|benzoyl|tretinoin|niacinamide"
    r"|dexameth"          # 항생제-스테로이드 복합 점안/점이제 (ciprofloxacin-dexamethasone 등)
    r"|\d%"               # 농도 표기 제형: 점안액(0.3%, 0.5%), 외용액(1%), lock 용액(sodium citrate 4%)
)

# route로 판단하는 비전신 투여
NON_SYSTEMIC_ROUTES = {
    "TP", "TD",                                          # topical / transdermal
    "BOTH EYES", "RIGHT EYE", "LEFT EYE", "OU", "OD", "OS",
    "BOTH EARS", "RIGHT EAR", "LEFT EAR", "AU", "AD", "AS",
    "INTRAVITREAL", "IVT",                               # 안구 내 주사
    "VG",                                                # vaginal
    "LOCK", "DWELL",                                     # catheter lock
    "IRR", "IR",                                         # irrigation (세척)
    "EX-VIVO",
}

_AGENT_RES = [(re.compile(p), agent) for p, agent in ANTIBIOTIC_AGENTS]
_NON_SYSTEMIC_RE = re.compile(NON_SYSTEMIC_NAME_PATTERN)


def map_antibiotic_agent(drug_name):
    """약명 -> 표준 항생제 성분명. 항생제가 아니거나 비전신 제형이면 None."""
    if not isinstance(drug_name, str):
        return None
    name = drug_name.lower().strip()
    if _NON_SYSTEMIC_RE.search(name):
        return None
    for pattern, agent in _AGENT_RES:
        if pattern.search(name):
            return agent
    return None
