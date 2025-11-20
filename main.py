# backend/main.py
from datetime import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import tempfile
import subprocess
import zipfile
import nibabel as nib
import numpy as np
import xgboost as xgb
import pymysql
from typing import List
from scipy.ndimage import label as cc_label

app = FastAPI()

# CORS 설정
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/api/process_mri")
async def options_handler():
    return JSONResponse(status_code=200)


# MySQL 설정
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "1111",
    "db": "memora_db",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

# PATH / 환경 설정
MODEL_PATH = Path("models/xgb_hippo.json")
DCM2NIIX_BIN = "dcm2niix"

# WSL / Conda 설정 (WSL의 기본 distro 사용)
HIPPMAPP3R_ENV_NAME = "hippmapp3r"
CONDA_INIT = "source ~/miniconda3/etc/profile.d/conda.sh"

# XGBoost 모델 로드
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(MODEL_PATH)

FEATURE_COLS = [
    "left_hipp_vol_mm3",
    "right_hipp_vol_mm3",
    "total_hipp_vol_mm3",
    "asymmetry_index",
    "left_hipp_vol_icv_norm",
    "right_hipp_vol_icv_norm",
    "total_hipp_vol_icv_norm",
    "AGE",
    "APOE4",
    "SEX_FEMALE",
]


# API 반환 모델
class ProcessResult(BaseModel):
    label: str
    probs: dict
    summary: str
    features: dict

# 환자 목록
class PatientOut(BaseModel):
    patient_id: str
    name: str
    sex: str
    age: int


# 공통 함수
def save_file_tmp(upload: UploadFile) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    dest = tmp_dir / upload.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def dicom_to_nifti(zip_path: Path) -> Path:
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
    p = Path(path)
    drive = p.drive[0].lower()  # C:\ -> c
    rest = str(p).replace(p.drive, "").replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def run_hippmapp3r(nii: Path) -> Path:
    """
    세그멘테이션만 WSL(Ubuntu)에서 HippMapp3r CLI(hippmapper seg_hipp)로 실행.
    QC 단계(ANTs ConvertScalarImageToRGB 실패)는 무시하고,
    pred.nii.gz가 있으면 성공으로 간주.
    """
    out_dir = nii.parent / "hippmapp3r"
    out_dir.mkdir(exist_ok=True)

    pred_path = out_dir / "pred.nii.gz"

    nii_wsl = win_to_wsl(nii)
    pred_wsl = win_to_wsl(pred_path)

    cmd = (
        f"{CONDA_INIT} && "
        f"conda activate {HIPPMAPP3R_ENV_NAME} && "
        f"hippmapper seg_hipp -t1 {nii_wsl} -o {pred_wsl}"
    )

    full_cmd = ["wsl", "bash", "-lc", cmd]

    print("=== WSL CMD ===")
    print(" ".join(full_cmd))
    print("================")

    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
    )

    print("=== WSL STDOUT ===")
    print(result.stdout)
    print("=== WSL STDERR ===")
    print(result.stderr)

    # 1) 세그멘테이션 결과 파일이 아예 없으면 진짜 실패로 취급
    if not pred_path.exists():
        raise RuntimeError(
            f"HippMapp3r(WSL) 실행 실패 (pred.nii.gz 없음, code={result.returncode}): {result.stderr}"
        )

    # 2) pred.nii.gz는 있는데, QC 때문에 returncode != 0 이면 경고만 찍고 계속 진행
    if result.returncode != 0:
        print(
            "⚠ HippMapp3r가 비정상 종료(returncode != 0) 했지만 "
            "pred.nii.gz는 생성됨. QC(ANTs ConvertScalarImageToRGB) 단계 오류로 추정, 무시하고 계속 진행."
        )

    return pred_path



def largest_cc(mask):
    labeled, comp = cc_label(mask)
    if comp == 0:
        return mask
    sizes = [(labeled == i).sum() for i in range(1, comp + 1)]
    best = np.argmax(sizes) + 1
    return (labeled == best).astype(np.uint8)


def split_left_right(pred_path: Path):
    img = nib.load(str(pred_path))
    data = img.get_fdata()

    L = largest_cc((data == 1).astype(np.uint8))
    R = largest_cc((data == 2).astype(np.uint8))

    left = pred_path.parent / "left.nii.gz"
    right = pred_path.parent / "right.nii.gz"

    nib.save(nib.Nifti1Image(L, img.affine), left)
    nib.save(nib.Nifti1Image(R, img.affine), right)

    return left, right


# compute_features
def compute_features(left_path: Path, right_path: Path, icv: float | None):
    imgL = nib.load(str(left_path))
    imgR = nib.load(str(right_path))

    L = imgL.get_fdata()
    R = imgR.get_fdata()
    vox = np.prod(imgL.header.get_zooms())

    volL = L.sum() * vox
    volR = R.sum() * vox
    volT = volL + volR

    asym = (volL - volR) / volT if volT > 0 else 0

    feats = {
        "left_hipp_vol_mm3": round(volL),
        "right_hipp_vol_mm3": round(volR),
        "total_hipp_vol_mm3": round(volT),
        "asymmetry_index": round(asym, 4),
        "left_hipp_vol_icv_norm": None,
        "right_hipp_vol_icv_norm": None,
        "total_hipp_vol_icv_norm": None,
        "icv": icv,
    }

    if icv and icv > 0:
        scale = 1000.0 / icv
        feats["left_hipp_vol_icv_norm"] = round(volL * scale, 3)
        feats["right_hipp_vol_icv_norm"] = round(volR * scale, 3)
        feats["total_hipp_vol_icv_norm"] = round(volT * scale, 3)

    return feats


# XGBoost
def build_vec(feats):
    return np.array([float(feats.get(c, 0)) for c in FEATURE_COLS], dtype=np.float32).reshape(1, -1)


def infer(feats):
    p = xgb_model.predict_proba(build_vec(feats))[0]
    raw = list(xgb_model.classes_)
    mapped = ["CN" if c in [0, "0"] else "AD" for c in raw]
    probs = {c: float(pp * 100) for c, pp in zip(mapped, p)}
    return max(probs, key=probs.get), probs


def make_summary(label, probs, feats):
    CN = round(probs["CN"])
    AD = round(probs["AD"])
    guide = "정상 범위로 예측됨." if label == "CN" else "알츠하이머 가능성이 높음."
    return (
        f"AI 예측: {label}\n"
        f"{guide}\n\n"
        f"확률: CN {CN}% · AD {AD}%\n"
        f"총 해마 부피: {feats['total_hipp_vol_mm3']} mm³"
    )


# DB 저장
def save_db(pid, filename, feats, label, probs, exam_id):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mri_results (
                    patient_id, label,
                    prob_cn, prob_ad,
                    left_hipp_vol, right_hipp_vol, total_hipp_vol,
                    icv, age, sex, apoe4,
                    filename,
                    exam_id
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    pid,
                    label,
                    probs["CN"],
                    probs["AD"],
                    feats["left_hipp_vol_mm3"],
                    feats["right_hipp_vol_mm3"],
                    feats["total_hipp_vol_mm3"],
                    feats["icv"],
                    feats["AGE"],
                    "F" if feats["SEX_FEMALE"] else "M",
                    feats["APOE4"],
                    filename,
                    exam_id,
                ),
            )
        conn.commit()
    finally:
        conn.close()

@app.get("/api/patients", response_model=List[PatientOut])
def get_patients():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    patient_id,
                    name,
                    sex,
                    TIMESTAMPDIFF(
                        YEAR,
                        STR_TO_DATE(
                            CONCAT(birth_year, '-', birth_month, '-', birth_day),
                            '%Y-%m-%d'
                        ),
                        CURDATE()
                    ) AS age
                FROM patients
                ORDER BY patient_id
                """
            )
            rows = cur.fetchall()
        return rows
    finally:
        conn.close()


# 메인 API
@app.post("/api/process_mri", response_model=ProcessResult)
async def process_mri(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    age: float = Form(...),
    apoe4: int = Form(...),
    sex: str = Form(...),
    icv: float | None = Form(None),
    exam_id: str | None = Form(None)
):
    if not exam_id:
        exam_id = f"EXAM_{int(time.time())}"
        
    tmp = save_file_tmp(file)

    # ZIP → DICOM 변환
    if tmp.suffix.lower() == ".zip":
        nii = dicom_to_nifti(tmp)
    else:
        nii = tmp

    # segmentation (WSL + HippMapp3r CLI)
    pred = run_hippmapp3r(nii)

    # left/right 생성
    left, right = split_left_right(pred)

    # feature 계산
    feats = compute_features(left, right, icv)
    feats["AGE"] = age
    feats["APOE4"] = apoe4
    feats["SEX_FEMALE"] = 1.0 if sex.upper().startswith("F") else 0.0

    # XGBoost
    label, probs = infer(feats)
    summary = make_summary(label, probs, feats)

    # DB
    save_db(patient_id, file.filename, feats, label, probs, exam_id)

    return JSONResponse(
        ProcessResult(
            label=label,
            probs=probs,
            summary=summary,
            features=feats,
        )
    )