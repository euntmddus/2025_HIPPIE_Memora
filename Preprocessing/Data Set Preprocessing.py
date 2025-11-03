import pandas as pd
from pathlib import Path

# 0) 경로
ROOT = Path(r"D:\ADNI_Project")
LABEL_DIR = Path(r"D:\ADNI_Project\metadata") 
LABEL_DIR.mkdir(exist_ok=True)

# 1) ADNIMERGE 불러오기 & 기본 전처리
src_path = ROOT / "ADNIMERGE_23Sep2025.csv"
df = pd.read_csv(src_path, low_memory=False)

# 날짜 파싱 (sc 가장 빠른 날짜로 선택)
if "EXAMDATE" in df.columns:
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
else:
    df["EXAMDATE"] = pd.NaT

# Whole brain 컬럼명 표준화
if "Whole brain" in df.columns and "WholeBrain" not in df.columns:
    df = df.rename(columns={"Whole brain": "WholeBrain"})==0-0-0

# VISCODE/PTID 정규화 
df["VISCODE"] = df["VISCODE"].astype(str).str.strip().str.lower()
df["PTID"] = df["PTID"].astype(str).str.strip().str.upper()

# 2) 사용 행 필터링: CN/AD 포함
use = df.copy()
use["DX_bl"] = use["DX_bl"].astype(str).str.strip().str.upper()
use = use[use["DX_bl"].isin(["CN", "AD"])]

# 3) 조건 컬럼 결측 제거 (요구 조건 충족 보장)
cond_cols = ["Hippocampus", "WholeBrain"]
missing_cond = [c for c in cond_cols if c not in use.columns]
if missing_cond:
    raise ValueError(f"조건 컬럼 누락: {missing_cond}")

# 수치화 및 결측 제거
for c in cond_cols:
    use[c] = pd.to_numeric(use[c], errors="coerce")
use = use.dropna(subset=cond_cols)

# 중간 저장 (라벨링 단계에서 그대로 사용)
pre_path = LABEL_DIR / "pre_ADNIMERGE.csv"
use.to_csv(pre_path, index=False)
