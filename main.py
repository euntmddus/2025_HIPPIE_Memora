# backend/main.py
# 정리: 상수/설정 -> 유틸(파일/변환) -> ICV/마스크 처리 -> 해마 특성 계산 -> 모델 추론 -> DB 관련 -> API 엔드포인트
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
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
import base64
from typing import List, Optional
from scipy.ndimage import label as cc_label
import json
import plotly.graph_objects as go
from skimage import measure
from scipy.ndimage import gaussian_filter
# from scipy.ndimage import binary_closing      # currently unused -> commented
# from skimage.filters import gaussian           # currently unused -> commented
from nilearn.masking import compute_brain_mask
from nilearn.image import resample_to_img

app = FastAPI()

# CORS 설정 (개발용)
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

# ------------------------------
# 상수 / 경로 / 모델 로드
# ------------------------------
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "1111",
    "db": "memora_db",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

MODEL_PATH = Path("models/xgb_hippo.json")
DCM2NIIX_BIN = "dcm2niix"

HIPPMAPP3R_ENV_NAME = "hippmapp3r"
CONDA_INIT = "source ~/miniconda3/etc/profile.d/conda.sh"

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# XGBoost 모델 로드 (한 번만 로드)
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

# ------------------------------
# Pydantic 응답/요청 모델
# ------------------------------
class ProcessResult(BaseModel):
    label: str
    probs: dict
    summary: str
    features: dict
    mask_base64: str | None = None
    exam_datetime: str | None = None
    exam_id: int | None = None

class PatientOut(BaseModel):
    patient_id: str
    name: str
    sex: str
    age: int
    height_cm: int | None = None
    weight_kg: int | None = None
    icv: float | None = None 
    apoe4: Optional[int] = None

class MaskRequest(BaseModel):
    mask_base64: str

class ExamHistoryItem(BaseModel):
    exam_id: int
    exam_datetime: str
    label: str
    total_hipp_vol: int
    created_at: str

# ------------------------------
# 파일 / 변환 유틸 함수
# ------------------------------
def save_file_permanent(upload: UploadFile) -> Path:
    "업로드 파일을 uploads 디렉토리에 영구 저장"
    dest = UPLOAD_DIR / upload.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest

def dicom_to_nifti(zip_path: Path) -> Path:
    "zip 내 DICOM을 dcm2niix로 NIfTI 변환 (출력 파일 반환)"
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
    "Windows 경로를 WSL 경로로 변환 (WSL에서 hippmapp3r 호출 시 사용)"
    p = Path(path).resolve() 
    drive = p.drive[0].lower() if p.drive else ''
    rest = str(p).replace(p.drive, "").replace("\\", "/")
    return f"/mnt/{drive}{rest}"

# ------------------------------
# HippMapp3r 실행 및 마스크 후처리
# ------------------------------
def run_hippmapp3r(nii: Path) -> Path:
    "WSL로 HippMapp3r를 실행하고 pred.nii.gz 반환"
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

    result = subprocess.run(full_cmd, capture_output=True, text=True)

    print("=== WSL STDOUT ===")
    print(result.stdout)
    print("=== WSL STDERR ===")
    print(result.stderr)

    if not pred_path.exists():
        raise RuntimeError(
            f"HippMapp3r(WSL) 실행 실패 (pred.nii.gz 없음, code={result.returncode}): {result.stderr}"
        )

    if result.returncode != 0:
        print(
            "⚠ HippMapp3r가 비정상 종료(returncode != 0) 했지만 "
            "pred.nii.gz는 생성됨. QC 단계 오류로 추정, 무시하고 계속 진행."
        )

    return pred_path

def largest_cc(mask):
    "binary mask의 가장 큰 connected component만 반환"
    labeled, comp = cc_label(mask)
    if comp == 0:
        return mask
    sizes = [(labeled == i).sum() for i in range(1, comp + 1)]
    best = np.argmax(sizes) + 1
    return (labeled == best).astype(np.uint8)

def split_left_right(pred_path: Path):
    "pred.nii.gz에서 라벨 1(L)과 2(R)을 분리하여 left.nii.gz, right.nii.gz 저장"
    img = nib.load(str(pred_path))
    data = img.get_fdata()

    L = largest_cc((data == 1).astype(np.uint8))
    R = largest_cc((data == 2).astype(np.uint8))

    left = pred_path.parent / "left.nii.gz"
    right = pred_path.parent / "right.nii.gz"

    nib.save(nib.Nifti1Image(L, img.affine), left)
    nib.save(nib.Nifti1Image(R, img.affine), right)

    return left, right

# ------------------------------
# ICV 계산
# ------------------------------
def calculate_icv_nilearn(nifti_path: Path) -> float:
    "nilearn의 compute_brain_mask로 ICV(mm3) 계산"
    try:
        print(f" Nilearn ICV 계산 시작: {nifti_path}")
        img = nib.load(str(nifti_path))
        mask_img = compute_brain_mask(img, threshold=0.5, connected=True)
        mask_data = mask_img.get_fdata()
        voxel_count = np.sum(mask_data > 0)
        header = img.header
        zooms = header.get_zooms()
        one_voxel_vol = zooms[0] * zooms[1] * zooms[2]
        icv_mm3 = voxel_count * one_voxel_vol
        print(f" ICV 계산 완료: {icv_mm3:.2f} mm³")
        return float(icv_mm3)
    except Exception as e:
        print(f" ICV 계산 실패: {e}")
        return 0.0

# ------------------------------
# 해마 특성 계산
# ------------------------------
def compute_features(left_path: Path, right_path: Path, icv: float | None):
    "Left/Right 마스크로부터 부피 및 ICV 정규화 지표 계산"
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

# ------------------------------
# 모델: 벡터 생성 및 추론
# ------------------------------
def build_vec(feats):
    "FEATURE_COLS 순서에 따라 입력 벡터 생성"
    vector = []
    for c in FEATURE_COLS:
        val = feats.get(c)
        if val is None:
            vector.append(np.nan)
        else:
            vector.append(float(val))
    return np.array(vector, dtype=np.float32).reshape(1, -1)

def infer(feats):
    "모델로 예측(확률) 반환"
    p = xgb_model.predict_proba(build_vec(feats))[0]
    raw = list(xgb_model.classes_)
    mapped = ["CN" if c in [0, "0"] else "AD" for c in raw]
    probs = {c: float(pp * 100) for c, pp in zip(mapped, p)}
    return max(probs, key=probs.get), probs

def make_summary(label, probs, feats):
    "간단한 결과 요약 문자열 생성"
    CN = round(probs["CN"])
    AD = round(probs["AD"])
    guide = "정상 범위로 예측됨." if label == "CN" else "알츠하이머 가능성이 높음."
    return (
        f"모델 예측: {label}\n"
        f"{guide}\n\n"
        f"확률: CN {CN}% · AD {AD}%\n"
        f"총 해마 부피: {feats.get('total_hipp_vol_mm3', '—')} mm³"
    )

# ------------------------------
# DB 관련 함수
# ------------------------------
def save_exam(patient_id, exam_dt):
    "mri_exams에 검사 기록을 저장하고 exam_id 반환"
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO mri_exams (patient_id, exam_datetime) VALUES (%s, %s)",
                (patient_id, exam_dt)
            )
            exam_id = cur.lastrowid 
        conn.commit()
        return exam_id
    finally:
        conn.close()

def save_db(pid, filename, feats, label, probs, exam_id):
    "mri_results 테이블에 결과 행 저장"
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mri_results 
                (
                    patient_id, label, prob_cn, prob_ad, 
                    left_hipp_vol, right_hipp_vol, total_hipp_vol, 
                    icv, age, sex, apoe4, filename, exam_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    feats.get("AGE"),           
                    "F" if feats.get("SEX_FEMALE") else "M",
                    feats.get("APOE4"),
                    filename,
                    exam_id
                ),
            )
        conn.commit()
    finally:
        conn.close()

# ------------------------------
# API 엔드포인트
# ------------------------------
@app.options("/api/process_mri")
async def options_handler():
    "CORS preflight 처리"
    return JSONResponse(status_code=200)

@app.get("/api/patients", response_model=List[PatientOut])
def get_patients():
    "환자 목록 반환"
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    patient_id,
                    name,
                    sex,
                    height_cm,
                    weight_kg,
                    icv, apoe4,
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

@app.post("/api/process_mri", response_model=ProcessResult)
async def process_mri(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    age: float | None = Form(None), 
    apoe4: Optional[int] = Form(None),
    sex: str | None = Form(None),
    icv: float | None = Form(None),
    exam_datetime: str | None = Form(None)
):
    "메인 파이프라인: 파일 저장 -> (DICOM->NIfTI) -> ICV 계산 -> HippMapp3r -> 특징 계산 -> 예측 -> DB저장 -> 마스크 base64 반환"
    # 1. 파일 저장 및 변환
    nii = save_file_permanent(file)
    if nii.suffix.lower() == ".zip":
        nii = dicom_to_nifti(nii)

    # 2. exam datetime 처리
    if exam_datetime:
        try:
            dt_obj = datetime.strptime(exam_datetime, '%Y-%m-%d %H:%M:%S')
        except:
            try:
                dt_obj = datetime.fromisoformat(exam_datetime.replace('T', ' '))
            except:
                dt_obj = datetime.now()
    else:
        dt_obj = datetime.now()

    # 3. ICV 계산 (수동값 없으면 자동 계산)
    final_icv = icv
    if final_icv is None or final_icv == 0:
        print(" ICV 정보가 없어 자동 계산을 시작합니다...")
        calculated_icv = calculate_icv_nilearn(nii)
        if calculated_icv > 0:
            final_icv = calculated_icv
        else:
            print("⚠️ ICV 계산 실패, 기본값 0.0 사용")
            final_icv = 0.0
    print(f" 최종 적용 ICV: {final_icv}")

    # 4. 해마 세그멘테이션 실행 및 좌우 분리
    pred = run_hippmapp3r(nii)
    left, right = split_left_right(pred)

    # 5. Feature 계산 및 메타데이터 결합
    feats = compute_features(left, right, final_icv)
    feats["AGE"] = age 
    feats["APOE4"] = apoe4
    if sex is not None:
        feats["SEX_FEMALE"] = 1.0 if sex.upper().startswith("F") else 0.0
    else:
        feats["SEX_FEMALE"] = None

    # 6. AI 예측 및 요약
    label, probs = infer(feats)
    summary = make_summary(label, probs, feats)

    # 7. DB 저장 (exam 추가 후 결과 저장)
    exam_id = save_exam(patient_id, dt_obj)
    save_db(patient_id, file.filename, feats, label, probs, exam_id)

    # 8. 마스크 파일을 원본 공간으로 리샘플링 후 base64 인코딩 반환
    mask_b64 = None
    try:
        if pred.exists():
            # (1) 원본과 예측 마스크 로드
            orig_img = nib.load(str(nii))
            pred_img = nib.load(str(pred))
            
            print(f"Mask 정합성 확보를 위한 리샘플링 수행...")
            
            # (2) 원본 MRI 그리드에 맞춰 마스크 리샘플링 (위치 보정 핵심)
            # interpolation='nearest'를 써야 0, 1, 2 라벨이 유지됨
            resampled_img = resample_to_img(pred_img, orig_img, interpolation='nearest')
            
            # (3) 데이터 추출 및 uint8(0~255) 변환
            # 리샘플링 결과가 float일 수 있으므로 반올림 후 정수형으로 변환
            # 이렇게 해야 용량도 줄고 프론트엔드에서 파싱할 때 오류가 없음
            resampled_data = np.round(resampled_img.get_fdata()).astype(np.uint8)

            # (4) [중요] 원본 이미지의 Affine(좌표 정보)을 사용하여 새 NIfTI 생성
            # 이렇게 하면 원본과 물리적으로 100% 동일한 공간을 갖게 됨
            final_mask_img = nib.Nifti1Image(resampled_data, orig_img.affine)
            
            # 헤더에도 데이터 타입 명시
            final_mask_img.header.set_data_dtype(np.uint8)
            
            # (5) 덮어쓰기
            nib.save(final_mask_img, str(pred))
            print("리샘플링, 타입 변환(uint8), 좌표 교정 완료.")

            # (6) Base64 인코딩
            with open(pred, "rb") as f:
                data_bytes = f.read()
                mask_b64 = base64.b64encode(data_bytes).decode('utf-8')
        else:
            print(">>> DEBUG: pred file does NOT exist:", pred)
            mask_b64 = None
    except Exception as e:
        print(">>> ERROR reading/resampling pred file:", e)
        import traceback
        traceback.print_exc() # 에러 상세 출력
        mask_b64 = None
        
    # 9. 결과 반환
    return JSONResponse(
        ProcessResult(
            label=label,
            probs=probs,
            summary=summary,
            features=feats,
            mask_base64=mask_b64,
            exam_datetime=dt_obj.strftime('%Y-%m-%d %H:%M:%S'),
            exam_id=exam_id
        ).dict()
    )

@app.get("/api/exams/{exam_id}")
def get_exam_detail(exam_id: int):
    "단일 exam 결과 상세 조회"
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT * FROM mri_results 
                WHERE exam_id = %s
            """
            cur.execute(sql, (exam_id,))
            row = cur.fetchone()
            if not row:
                return JSONResponse({"status": "error", "message": "데이터 없음"}, status_code=404)

            feats = {
                "icv": row['icv'],
                "left_hipp_vol_mm3": row['left_hipp_vol'],
                "right_hipp_vol_mm3": row['right_hipp_vol'],
                "total_hipp_vol_mm3": row['total_hipp_vol'],
                "APOE4": row['apoe4'],
                "left_hipp_vol_icv_norm": None,
                "right_hipp_vol_icv_norm": None,
                "total_hipp_vol_icv_norm": None
            }

            if row['icv'] and row['icv'] > 0:
                scale = 1000.0 / row['icv']
                feats["left_hipp_vol_icv_norm"] = round(row['left_hipp_vol'] * scale, 3)
                feats["right_hipp_vol_icv_norm"] = round(row['right_hipp_vol'] * scale, 3)
                feats["total_hipp_vol_icv_norm"] = round(row['total_hipp_vol'] * scale, 3)

            probs = {"CN": row['prob_cn'], "AD": row['prob_ad']}
            summary = make_summary(row['label'], probs, feats)

            file_url = f"http://127.0.0.1:8000/uploads/{row['filename']}"

            return {
                "status": "success",
                "data": {
                    "file_url": file_url,
                    "filename": row['filename'],
                    "features": feats,
                    "summary": summary,
                    "label": row['label'],
                    "probs": probs
                }
            }
    finally:
        conn.close()

@app.get("/api/patients/{patient_id}/history", response_model=List[ExamHistoryItem])
def get_patient_history(patient_id: str):
    "환자별 검사 이력 반환"
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    r.exam_id,
                    DATE_FORMAT(e.exam_datetime, '%%Y-%%m-%%d %%H:%%i:%%s') as exam_datetime,
                    r.label,
                    r.total_hipp_vol,
                    DATE_FORMAT(r.created_at, '%%Y-%%m-%%d %%H:%%i:%%s') as created_at
                FROM mri_results r
                JOIN mri_exams e ON r.exam_id = e.id
                WHERE r.patient_id = %s
                ORDER BY e.exam_datetime DESC
            """
            cur.execute(query, (patient_id,))
            rows = cur.fetchall()
            history = []
            for row in rows:
                history.append(ExamHistoryItem(
                    exam_id=row['exam_id'],
                    exam_datetime=str(row['exam_datetime']),
                    label=row['label'],
                    total_hipp_vol=row['total_hipp_vol'],
                    created_at=str(row['created_at'])
                ))
            return history
    except pymysql.MySQLError as e:
        print("DB error in get_patient_history:", e)
        return JSONResponse({"status": "error", "message": "DB error: " + str(e)}, status_code=500)
    finally:
        conn.close()

@app.post("/api/get_plotly_3d")
async def get_plotly_3d(req: MaskRequest):
    "mask_base64를 받아 Plotly 3D mesh JSON을 생성하여 반환"
    try:
        print(">>> [3D 최종] 홈 버튼 고정 + 카메라 90도 회전 + 라벨/축 표시 + 고화질 렌더링")
        import base64, gzip
        decoded = base64.b64decode(req.mask_base64)
        if decoded[:2] == b'\x1f\x8b':
            decoded = gzip.decompress(decoded)

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name

        img = nib.load(tmp_path)
        data = np.round(img.get_fdata()).astype(int)
        Path(tmp_path).unlink()

        if np.sum(data > 0) < 10:
            return JSONResponse({"status": "error", "message": "해마 데이터가 없습니다."}, status_code=400)

        coords = np.argwhere(data > 0)
        z_mean = np.mean(coords[:, 0])
        y_mean = np.mean(coords[:, 1])
        x_mean = np.mean(coords[:, 2])
        center = np.array([z_mean, y_mean, x_mean])

        traces = []
        label_positions = {}
        all_x, all_y, all_z = [], [], []

        def create_trace(label_id, color, name, text_label):
            "label_id에 해당하는 볼륨을 추출하여 Mesh3d trace 생성"
            if np.sum(data == label_id) < 10: return None
            m = (data == label_id).astype(float)
            m_smooth = gaussian_filter(m, sigma=0.5) # 값이 클 수록 매끈매끈
            try:
                verts, faces, _, _ = measure.marching_cubes(m_smooth, level=0.5, step_size=1)
                verts_xyz = np.vstack([verts[:, 2], verts[:, 1], verts[:, 0]]).T
                verts_centered = verts_xyz - np.array([center[2], center[1], center[0]])
                all_x.extend(verts_centered[:, 0])
                all_y.extend(verts_centered[:, 1])
                all_z.extend(verts_centered[:, 2])
                label_positions[text_label] = {
                    "x": np.mean(verts_centered[:, 0]),
                    "y": np.mean(verts_centered[:, 1]),
                    "z": np.mean(verts_centered[:, 2]),
                    "color": color
                }
                return go.Mesh3d(
                    x=verts_centered[:, 0].tolist(),
                    y=verts_centered[:, 1].tolist(),
                    z=verts_centered[:, 2].tolist(),
                    i=faces[:, 0].tolist(),
                    j=faces[:, 1].tolist(),
                    k=faces[:, 2].tolist(),
                    color=color,
                    opacity=1.0,
                    name=name,
                    flatshading=False,
                    lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.7, specular=0.1, fresnel=0.5),
                    lightposition=dict(x=1000, y=1000, z=5000)
                )
            except Exception as e:
                print(f"메쉬 오류 ({name}): {e}")
                return None

        t1 = create_trace(1, '#27ae60', 'Left Hippocampus', "L")
        if t1: traces.append(t1)
        t2 = create_trace(2, '#e74c3c', 'Right Hippocampus', "R")
        if t2: traces.append(t2)

        if not traces:
            return JSONResponse({"status": "error", "message": "3D 모델 생성 실패"}, status_code=500)

        fig = go.Figure(data=traces)

        annotations = []
        for txt, pos in label_positions.items():
            annotations.append(dict(
                showarrow=False,
                x=pos["x"], y=pos["y"], z=pos["z"] + 15,
                text=txt,
                font=dict(color=pos["color"], size=24, family="Arial Black"),
                xanchor="center", yanchor="bottom"
            ))

        padding = 20
        if all_x:
            max_x, max_y, max_z = max(all_x), max(all_y), max(all_z)
            annotations.append(dict(showarrow=False, x=max_x + padding, y=0, z=0, text="x", font=dict(color="black", size=14)))
            annotations.append(dict(showarrow=False, x=0, y=max_y + padding, z=0, text="y", font=dict(color="black", size=14)))
            annotations.append(dict(showarrow=False, x=0, y=0, z=max_z + padding, text="z", font=dict(color="black", size=14)))

        axis_style = dict(
            showgrid=False, zeroline=False, showbackground=False, showticklabels=False,
            visible=False, showline=False
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(**axis_style),
                yaxis=dict(**axis_style),
                zaxis=dict(**axis_style),
                aspectmode='data',
                bgcolor='white',
                annotations=annotations,
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            paper_bgcolor='white',
            margin=dict(l=0, r=0, b=0, t=0)
        )

        return JSONResponse(content=json.loads(fig.to_json()))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# ------------------------------
# 함수 순서 요약 (읽기 편한 흐름)
# 1) 설정/상수
# 2) 파일/변환 유틸(save_file_permanent, dicom_to_nifti, win_to_wsl)
# 3) segmentation 실행(run_hippmapp3r) 및 마스크 후처리(split_left_right, largest_cc)
# 4) ICV 계산(calculate_icv_nilearn)
# 5) 특성 계산(compute_features)
# 6) 모델 관련(build_vec, infer, make_summary)
# 7) DB 관련(save_exam, save_db)
# 8) API 엔드포인트 (/api/process_mri 등)
# ------------------------------