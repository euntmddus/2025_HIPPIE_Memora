import os, re, glob, shutil, argparse, json, warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import nibabel as nib
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

SUPPORTED_IMG_SUFFIXES = [
    "_T1w.nii.gz", "_T1w.nii",
    "_acq-MPRAGE_T1w.nii.gz", "_acq-MPRAGE_T1w.nii",
    "_run-1_T1w.nii.gz", "_run-1_T1w.nii",
]

def is_t1_candidate(name: str) -> bool:
    return any(name.endswith(s) for s in SUPPORTED_IMG_SUFFIXES)

def normalize_dx_text(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    t = str(s).strip().upper()
    if any(k in t for k in ["CN", "NL", "NORMAL", "CONTROL"]):
        return "NC"
    if "AD" in t or "DEMENTIA" in t:
        return "AD"
    if "MCI" in t:
        return "MCI"
    return None

def load_dx_table(clin_csv: Path) -> pd.DataFrame:
    if clin_csv.suffix.lower() == ".csv":
        df = pd.read_csv(clin_csv, low_memory=False)
    elif clin_csv.suffix.lower() in [".tsv", ".txt"]:
        df = pd.read_csv(clin_csv, sep="\t", low_memory=False)
    else:
        raise ValueError("임상 CSV/TSV 파일만 지원합니다.")
    
    cols = {c.lower().replace('_', '').replace('-', ''): c for c in df.columns}
    
    pid_candidates = ["participant_id", "participantid", "subject", "oasisid", "id", "ptid"]
    pid_col = next((cols[c] for c in pid_candidates if c in cols), None)
    
    session_candidates = ["session_id", "sessionid", "session", "ses", "visit"]
    session_col = next((cols[c] for c in session_candidates if c in cols), None)

    dx_candidates = ["dx", "diagnosis", "group", "cdr_diagnosis", "clinical_status"]
    dx_col = next((cols[c] for c in dx_candidates if c in cols), None)
    
    if pid_col is None or dx_col is None:
        raise ValueError(f"필수 컬럼(Participant ID 또는 Diagnosis)을 찾지 못했습니다. 사용 가능한 컬럼: {list(df.columns)}")
    
    print(f"Using columns - PID: {pid_col}, DX: {dx_col}, Session: {session_col}")
    
    base_cols = [pid_col, dx_col]
    if session_col:
        base_cols.append(session_col)
    
    base = df[base_cols].copy()
    base.columns = ['PID', 'DX'] + (['Session'] if session_col else [])
    
    base['PID'] = base['PID'].astype(str).str.strip()
    base['PID'] = base['PID'].apply(lambda s: f"sub-{s}" if not s.startswith("sub-") else s)
    
    base['DX'] = base['DX'].apply(normalize_dx_text)
    base = base.dropna(subset=['DX'])
    
    if session_col:
        base = base.sort_values(['PID', 'Session']).drop_duplicates('PID', keep='last')
    else:
        base = base.drop_duplicates('PID', keep='last')
    
    print(f"처리 후 {len(base)}명의 고유 환자 데이터 로드 완료.")
    print(f"진단 분포: {base['DX'].value_counts().to_dict()}")
    
    return base.set_index('PID')

def map_images_enhanced(oasis_root: Path) -> List[Tuple[str, str, Path]]:
    out = []
    nii_files = list(Path(oasis_root).rglob("*.nii.gz")) + list(Path(oasis_root).rglob("*.nii"))
    
    print(f"총 {len(nii_files)}개의 NIfTI 파일 검색됨...")
    
    session_pattern = re.compile(r"ses-([a-zA-Z0-9]+)")
    subject_pattern = re.compile(r"sub-([a-zA-Z0-9]+)")
    
    processed_subjects = defaultdict(list)

    for p in nii_files:
        if not is_t1_candidate(p.name):
            continue
        
        path_str = str(p)
        subject_match = subject_pattern.search(path_str)
        if not subject_match:
            continue
        pid = f"sub-{subject_match.group(1)}"
        
        session_match = session_pattern.search(path_str)
        session = f"ses-{session_match.group(1)}" if session_match else "ses-01"
        
        processed_subjects[pid].append((session, p))

    for pid, sessions in processed_subjects.items():
        if not sessions:
            continue
        sessions.sort(key=lambda x: x[0])
        last_session, img_path = sessions[-1]
        out.append((pid, last_session, img_path))

    print(f"{len(out)}개의 T1w 이미지 후보(환자별 최신 1개)를 찾았습니다.")
    return out

def quality_check_image(img_path: Path) -> Dict[str, Any]:
    try:
        img = nib.load(img_path)
        data = img.get_fdata()
        header = img.header
        
        shape = data.shape
        voxel_size = header.get_zooms()[:3]
        
        stats = {
            'shape': list(shape),
            'voxel_spacing': [float(f"{v:.2f}") for v in voxel_size],
            'has_nan': bool(np.any(np.isnan(data))),
            'has_inf': bool(np.any(np.isinf(data)))
        }
        
        quality_flags = []
        if stats['has_nan']: quality_flags.append("NaN_values")
        if stats['has_inf']: quality_flags.append("Inf_values")
        if any(s < 64 for s in shape): quality_flags.append("Small_dimensions")
        if any(v > 2.0 for v in voxel_size): quality_flags.append("Large_voxel_spacing")

        stats['quality_flags'] = quality_flags
        stats['quality_score'] = max(0, 100 - len(quality_flags) * 25)
        
        return stats
        
    except Exception as e:
        return {
            'shape': [0,0,0], 'voxel_spacing': [0,0,0],
            'quality_flags': ['Loading_error'],
            'quality_score': 0,
            'error': str(e)
        }

def main():
    ap = argparse.ArgumentParser(description="Enhanced OASIS-3 → Pipeline 매핑")
    ap.add_argument('--oasis_root', required=True, type=Path, help="OASIS-3 BIDS root directory")
    ap.add_argument('--clin_csv', required=True, type=Path, help="Clinical data CSV/TSV file")
    ap.add_argument('--out_images', required=True, type=Path, help="정리된 이미지가 저장될 디렉토리")
    ap.add_argument('--out_csv', required=True, type=Path, help="최종 생성될 cls.csv 파일 경로")
    ap.add_argument('--include_mci_as', choices=['exclude','nc','ad'], default='exclude', 
                   help="MCI 환자 처리 방식: 'exclude'(제외), 'nc'(정상군으로), 'ad'(환자군으로)")
    ap.add_argument('--op', choices=['copy','symlink','move'], default='copy',
                   help="파일 처리 방식: copy(복사), symlink(바로가기 생성), move(이동)")
    ap.add_argument('--quality_check', action='store_true', 
                   help="이미지 품질(QC) 검사 수행")
    ap.add_argument('--min_quality_score', type=float, default=50,
                   help="QC 통과를 위한 최소 점수 (0-100)")

    args = ap.parse_args()

    print("=== Enhanced OASIS-3 Dataset Mapping ===")
    
    print("\n1. Loading clinical data...")
    dx_table = load_dx_table(args.clin_csv)
    
    print("\n2. Mapping images...")
    image_mappings = map_images_enhanced(args.oasis_root)
    
    if not image_mappings:
        raise SystemExit("T1 NIfTI 후보를 찾지 못했습니다. 경로를 확인하세요.")
    
    args.out_images.mkdir(parents=True, exist_ok=True)
    
    print(f"\n3. Processing mappings (Operation: {args.op})...")
    records = []
    used_ids = set()
    
    for pid, session, img_path in image_mappings:
        if pid not in dx_table.index:
            print(f"  - Skipping {pid} (no clinical data found)")
            continue
        
        dxv = dx_table.loc[pid]['DX']
        
        label = None
        if dxv == 'MCI':
            if args.include_mci_as == 'exclude':
                continue
            label = 'NC' if args.include_mci_as == 'nc' else 'AD'
        else:
            label = dxv
            
        if label not in ['NC', 'AD']:
            continue

        qc_stats = {}
        if args.quality_check:
            qc_stats = quality_check_image(img_path)
            if qc_stats.get('quality_score', 100) < args.min_quality_score:
                print(f"  - Skipping {pid} (QC score {qc_stats.get('quality_score')}, flags: {qc_stats.get('quality_flags')})")
                continue

        out_id = f"{pid}_{session}"
        if out_id in used_ids:
            continue
        used_ids.add(out_id)
        
        dst = args.out_images / f"{out_id}.nii.gz"
        
        try:
            if args.op == 'copy':
                shutil.copy2(img_path, dst)
            elif args.op == 'symlink':
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink(os.path.abspath(img_path), dst)
            else: # move
                shutil.move(str(img_path), dst)
        except Exception as e:
            print(f"  - ERROR processing {pid}: {e}")
            continue

        record = {
            'id': out_id, 
            'path': str(dst), 
            'label': label
        }
        record.update(qc_stats)
        records.append(record)

    if not records:
        raise SystemExit("매칭된 샘플이 없습니다. --include_mci_as 옵션을 확인하세요.")

    final_df = pd.DataFrame(records).sort_values('id')
    final_df.to_csv(args.out_csv, index=False)
    
    print("\n" + "="*30)
    print(f"[DONE] cls.csv 저장 완료: {args.out_csv}")
    print(f"  - 총 샘플 수: {len(records)}")
    print(f"  - 최종 분포: {final_df['label'].value_counts().to_dict()}")
    print("="*30)

if __name__ == "__main__":
    main()