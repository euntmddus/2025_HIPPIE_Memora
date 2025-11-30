# backend/file_utils.py
from datetime import datetime
from pathlib import Path
from fastapi import UploadFile
import shutil
import zipfile
import subprocess
import uuid

from .config import UPLOAD_DIR, DCM2NIIX_BIN


def save_file_permanent(upload: UploadFile) -> Path:
    """
    업로드 1건당 uploads/하위에 전용 폴더를 만들고 그 안에 저장
    """
    UPLOAD_DIR.mkdir(exist_ok=True)

    original_name = Path(upload.filename).name  # 원본 파일명
    # 폴더 이름: 날짜_시간_랜덤
    unique_dir = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
    subdir = UPLOAD_DIR / unique_dir
    subdir.mkdir(parents=True, exist_ok=True)

    dest = subdir / original_name
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

    return dest


def dicom_to_nifti(zip_path: Path) -> Path:
    """
    DICOM zip → NIfTI 변환
    """
    unzip = zip_path.parent / "dicom_unzip"
    unzip.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(unzip)

    out_dir = zip_path.parent / "nifti"
    out_dir.mkdir(exist_ok=True)

    cmd = [DCM2NIIX_BIN, "-z", "y", "-f", "converted", "-o", str(out_dir), str(unzip)]
    subprocess.run(cmd, check=True)

    nii = list(out_dir.glob("*.nii*"))
    if not nii:
        raise RuntimeError("NIfTI 생성 실패")
    return nii[0]


def win_to_wsl(path: Path | str) -> str:
    """
    Windows 경로 → WSL 경로 변환
    """
    p = Path(path).resolve()
    drive = p.drive[0].lower() if p.drive else ""
    rest = str(p).replace(p.drive, "").replace("\\", "/")
    return f"/mnt/{drive}{rest}"
