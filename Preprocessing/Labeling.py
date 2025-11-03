import pandas as pd
from pathlib import Path

# 0) 경로
ROOT = Path(r"D:\ADNI_Project")
LABEL_DIR = Path(r"D:\ADNI_Project\metadata") 
LABEL_DIR.mkdir(exist_ok=True)

# 1) 전처리된 파일 불러오기
src_path = LABEL_DIR / "pre_ADNIMERGE.csv"
use = pd.read_csv(src_path, low_memory=False)

# 4) PTID별 3단계 선택
BL_SET = {"bl"} 
SC_SET = {"m00"}

use["is_bl"] = use["VISCODE"].isin(BL_SET)
use["is_sc"] = use["VISCODE"].isin(SC_SET)

use["prio"] = 2  
use.loc[use["is_sc"], "prio"] = 1 # prio=1: SC 그룹 (차선)
use.loc[use["is_bl"], "prio"] = 0 # prio=0: BL 그룹 (최우선)

# mxx 데이터 중에서 가장 이른 날짜로 폴백하는 로직 구현
use = use.sort_values(["PTID", "prio", "EXAMDATE"], ascending=[True, True, True])

# PTID 유니크 확보 (최적의 1건만 남음)
chosen = (use.drop_duplicates(subset=["PTID"], keep="first")
             .assign(PTID=lambda d: d["PTID"].astype(str).str.strip().str.upper()))

# 라벨 매핑
label_map = {"CN": 0, "AD": 1}
chosen["label"] = chosen["DX_bl"].map(label_map)

# 5) 저장
out_cols_save = ["PTID", "label"]
cls_save = chosen[out_cols_save].reset_index(drop=True)

out_path = LABEL_DIR / "cls.csv"
cls_save.to_csv(out_path, index=False)

# 6) 결과 출력
original_unique_ptids = use["PTID"].nunique()
final_df = chosen
cond_cols = ["Hippocampus", "WholeBrain"]

print(f"[전체 원본 데이터] 모든 방문 및 조건 포함, 고유 PTID 총 개수: {original_unique_ptids}개\n")
print(f"[라벨링 완료] 저장 경로: {out_path}")
print(f"- 총 PTID: {final_df['PTID'].nunique()}")
print(f"- 라벨 분포:\n{final_df['label'].value_counts(dropna=False)}")

vis = final_df["VISCODE"].astype(str).str.lower()
BL_SET = {"bl"}
SC_SET = {"m00"}

print("\n- VISCODE 선택 분포:")
viscode_counts = final_df["VISCODE"].value_counts().sort_index()
print(viscode_counts)

print("\n- 주요 컬럼 통계 (선택된 데이터):")
print(final_df[cond_cols].describe().T)
