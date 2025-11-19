# 2) 사용 행 필터링: 모든 VISCODE + CN/AD
use = df.copy()
use["DX_bl"] = use["DX_bl"].astype(str).str.strip().str.upper()
use = use[use["DX_bl"].isin(["CN", "AD"])]

# 라벨 매핑
label_map = {"CN": 0, "AD": 1}
use["label"] = use["DX_bl"].map(label_map)

# 4) PTID별 3단계 선택(mXX -> 중 가장 빠른 날짜 선택)
BL_SET = {"bl"}
SC_SET = {"m00"}

# 우선순위 플래그 및 Prio 컬럼 생성
use["is_bl"] = use["VISCODE"].isin(BL_SET)
use["is_sc"] = use["VISCODE"].isin(SC_SET)

# 후속 방문
use["prio"] = 2
use.loc[use["is_sc"], "prio"] = 1
use.loc[use["is_bl"], "prio"] = 0

use = use.sort_values(["PTID", "prio", "EXAMDATE"], ascending=[True, True, True])

# PTID 유니크 확보 (최적의 1건만 남음)
chosen = (use.drop_duplicates(subset=["PTID"], keep="first")
             .assign(PTID=lambda d: d["PTID"].astype(str).str.strip().str.upper()))

# 5) 저장
out_cols_save = ["PTID", "label"]
cls_save = chosen[out_cols_save].reset_index(drop=True)

out_path = LABEL_DIR / "cls.csv"
cls_save.to_csv(out_path, index=False)

original_unique_ptids = df['PTID'].nunique()
print(f"[전체 원본 데이터] 모든 방문 및 조건 포함, 고유 PTID 총 개수: {original_unique_ptids}개\n")

# 6) 결과 출력
final_df = chosen
vis = final_df["VISCODE"].astype(str).str.lower()
is_bl_sel = vis.isin(BL_SET)
is_sc_sel = vis.isin(SC_SET)

print(f"[라벨링 완료] 저장 경로: {out_path}")
print(f"- 총 PTID: {final_df['PTID'].nunique()}")
print(f"- 라벨 분포:\n{final_df['label'].value_counts(dropna=False)}")

print("\n- VISCODE 선택 분포:")
viscode_counts = final_df['VISCODE'].value_counts().sort_index()
print(viscode_counts)

print("\n- 주요 컬럼 통계 (선택된 데이터):")
print(final_df[cond_cols].describe().T)
