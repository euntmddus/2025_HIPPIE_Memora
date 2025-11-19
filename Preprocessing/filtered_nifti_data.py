import re
import pandas as pd
from pathlib import Path
import os

# 1. 경로 및 PTID 추출 로직 정의
ROOT = Path(r'D:\ADNI_Project')
RAW = ROOT / 'raw_nifti'
MD = ROOT / 'metadata'

# PTID 추출 정규식 강화
PTID_PATTERNS = [
    (re.compile(r"(\d{3})[_\-\s]?S[_\-\s]?(\d{4,5})", re.I), "A_fwd"),
    (re.compile(r"S[_\-\s]?(\d{4,5})[_\-\s]?(\d{3})", re.I), "B_rev"),
]
def norm_ptid(a, b):
    return f"{str(a).zfill(3)}_S_{str(b).zfill(4)}"

def extract_ptid(text: str):
    text = text.upper()
    for creg, tag in PTID_PATTERNS:
        m = creg.search(text)
        if m:
            if tag == "B_rev":
                return norm_ptid(m.group(2), m.group(1))
            return norm_ptid(m.group(1), m.group(2))
    return None

# 2. filemap.csv 생성
rows = []
for p in RAW.rglob('*.nii.gz'):
    ptid = extract_ptid(p.name)
    if not ptid:
        ptid = extract_ptid(str(p.parent))

   # PTID를 찾지 못했으면 상위 디렉토리명에서 백업 추출
    if ptid:
        rows.append({'path': str(p.relative_to(ROOT)).replace("\\", "/"), 
                     'filename': p.name, 
                     'PTID': ptid})
filemap = pd.DataFrame(rows)
filemap = filemap.dropna(subset=['PTID']).drop_duplicates('path')
filemap.to_csv(MD / 'filemap.csv', index=False)
print(f"filemap.csv 생성 완료. 총 NIfTI 파일 수: {len(filemap)}")

# 3. cls 사용 train_index.csv 생성
cls = pd.read_csv(MD / 'cls.csv')

# PTID를 기준으로 Inner Join
merged_df = filemap.merge(cls, on='PTID', how='inner')

# PTID별로 단 하나의 파일만 남김 (중복 제거)
train_index = merged_df.drop_duplicates(subset=['PTID'], keep='first')

# 저장
train_index = train_index[['path', 'PTID', 'label']]
train_index.to_csv(MD / 'train_index.csv', index=False)

print(f"\ntrain_index.csv 생성 완료.")
print(f"cls PTID 수 ({len(cls)}) 대비 매칭된 파일 수: {len(train_index)}")
