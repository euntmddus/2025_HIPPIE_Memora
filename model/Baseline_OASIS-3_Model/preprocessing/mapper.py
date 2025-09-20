"""
mapper.py — OASIS-3 → Pipeline 매핑 (Python 3.9, TF2.10 호환, TF 미사용)
- OASIS-3 BIDS(T1 NIfTI) + 임상표(CSV/TSV) → cls.csv(id,path,label) 생성
- 옵션: MCI 처리 정책(exclude/nc/ad), 파일 복사/링크/이동
"""
import os, re, glob, shutil, argparse
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd

SUPPORTED_IMG_SUFFIXES = [
    "_T1w.nii.gz", "_T1w.nii",
    "_acq-MPRAGE_T1w.nii.gz", "_acq-MPRAGE_T1w.nii",
]

def is_t1_candidate(name: str) -> bool:
    return any(name.endswith(s) for s in SUPPORTED_IMG_SUFFIXES)

def normalize_dx_text(s: str):
    if not isinstance(s, str):
        return None
    t = str(s).strip().upper()
    if any(k in t for k in ["CN","NL","NORMAL","CONTROL"]):
        return "NC"
    if "AD" in t or "DEMENTIA" in t:
        return "AD"
    if "MCI" in t:
        return "MCI"
    return None

def load_dx_table(clin_csv: Path) -> pd.DataFrame:
    if clin_csv.suffix.lower() == ".csv":
        df = pd.read_csv(clin_csv)
    elif clin_csv.suffix.lower() in [".tsv",".txt"]:
        df = pd.read_csv(clin_csv, sep="\t")
    else:
        raise ValueError("임상 CSV/TSV 파일만 지원합니다.")
    cols = {c.lower(): c for c in df.columns}
    pid = None; dx = None
    for k in ["participant_id","subject","oasisid","id"]:
        if k in cols: pid = cols[k]; break
    for k in ["dx","diagnosis","group","cdr_diagnosis"]:
        if k in cols: dx = cols[k]; break
    if pid is None or dx is None:
        raise ValueError("임상표에서 participant_id/diagnosis 컬럼을 찾지 못했습니다.")
    base = df[[pid, dx]].copy().rename(columns={pid:"PID", dx:"DX"})
    base["PID"] = base["PID"].astype(str).str.strip()
    if not base["PID"].str.startswith("sub-").any():
        base["PID"] = base["PID"].apply(lambda s: f"sub-{s}" if not s.startswith("sub-") else s)
    base["DX"] = base["DX"].apply(normalize_dx_text)
    base = base.drop_duplicates("PID")
    return base

def map_images(oasis_root: Path) -> List[Tuple[str, Path]]:
    out = []
    nii = list(Path(oasis_root).rglob("*.nii")) + list(Path(oasis_root).rglob("*.nii.gz"))
    for p in nii:
        if not is_t1_candidate(p.name):
            continue
        # BIDS 구조: sub-XXXX/ses-XX/anat/*
        parts = p.parts
        pid = None
        for part in parts:
            if part.startswith("sub-"):
                pid = part
                break
        if not pid:
            continue
        out.append((pid, p))
    return out

def main():
    ap = argparse.ArgumentParser(description="OASIS-3 → Pipeline 매핑")
    ap.add_argument('--oasis_root', required=True, type=Path)
    ap.add_argument('--clin_csv', required=True, type=Path)
    ap.add_argument('--out_images', required=True, type=Path)
    ap.add_argument('--out_csv', required=True, type=Path)
    ap.add_argument('--include_mci_as', choices=['exclude','nc','ad'], default='exclude')
    ap.add_argument('--op', choices=['copy','symlink','move'], default='copy')
    args = ap.parse_args()

    dx = load_dx_table(args.clin_csv)
    imgs = map_images(args.oasis_root)
    if not imgs:
        raise SystemExit("T1 NIfTI 후보를 찾지 못했습니다. 경로/명명 규칙을 확인하세요.")

    args.out_images.mkdir(parents=True, exist_ok=True)
    records = []
    used = set()

    for pid, path in imgs:
        row = dx[dx['PID'].astype(str) == str(pid)]
        if row.empty:
            continue
        dxv = row.iloc[0]['DX']
        if dxv is None:
            continue
        if dxv == 'MCI':
            if args.include_mci_as == 'exclude':
                continue
            elif args.include_mci_as == 'nc':
                label = 'NC'
            else:
                label = 'AD'
        else:
            label = dxv  # 'NC' or 'AD'

        stem = path.stem.replace('.nii','').replace('.gz','')
        out_id = f"{pid}_{stem}"
        if out_id in used:
            continue
        used.add(out_id)

        dst = args.out_images / f"{out_id}.nii.gz"
        if args.op == 'copy':
            shutil.copy2(path, dst)
        elif args.op == 'symlink':
            if dst.exists():
                dst.unlink()
            os.symlink(os.path.abspath(path), dst)
        else:
            shutil.move(str(path), dst)

        records.append({'id': out_id, 'path': str(dst), 'label': label})

    if not records:
        raise SystemExit("라벨과 매칭된 샘플이 생성되지 않았습니다. 옵션을 조정해보세요.")

    pd.DataFrame(records).sort_values('id').to_csv(args.out_csv, index=False)
    print(f"[DONE] cls.csv 저장: {args.out_csv} (n={len(records)})")

if __name__ == "__main__":
    main()
