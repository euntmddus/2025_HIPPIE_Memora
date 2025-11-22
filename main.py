# backend/main.py
from datetime import datetime
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
import base64
from typing import List, Optional
from scipy.ndimage import label as cc_label
import json
import plotly.graph_objects as go
from skimage import measure
from scipy.ndimage import binary_closing
from skimage.filters import gaussian

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
    mask_base64: str | None = None
    exam_datetime: str | None = None
    exam_id: int | None = None

# 환자 목록
class PatientOut(BaseModel):
    patient_id: str
    name: str
    sex: str
    age: int
    height_cm: int | None = None # 추가됨
    weight_kg: int | None = None # 추가됨
    icv: float | None = None 
    apoe4: int | None = 0

class MaskRequest(BaseModel):
    mask_base64: str
    
class ExamHistoryItem(BaseModel):
    exam_id: int
    exam_datetime: str
    label: str
    total_hipp_vol: int
    created_at: str

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
    return np.array([float(feats.get(c) or 0) for c in FEATURE_COLS], dtype=np.float32).reshape(1, -1)


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
        f"모델 예측: {label}\n"
        f"{guide}\n\n"
        f"확률: CN {CN}% · AD {AD}%\n"
        f"총 해마 부피: {feats['total_hipp_vol_mm3']} mm³"
    )

def save_exam(patient_id, exam_dt):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            # mri_exams 테이블에 날짜 저장
            cur.execute(
                "INSERT INTO mri_exams (patient_id, exam_datetime) VALUES (%s, %s)",
                (patient_id, exam_dt)
            )
            # 방금 저장된 행의 ID(exam_id)를 가져옴 (★핵심)
            exam_id = cur.lastrowid 
        conn.commit()
        return exam_id
    finally:
        conn.close()
        
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
                    filename, exam_id 
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
                    exam_id  # ★ 여기에 exam_id 추가됨
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


# 메인 API
@app.post("/api/process_mri", response_model=ProcessResult)
async def process_mri(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    age: float = Form(...),
    apoe4: int = Form(...),
    sex: str = Form(...),
    icv: float | None = Form(None),
    exam_datetime: str | None = Form(None) 
):
    # 1. 파일 저장 및 변환
    tmp = save_file_tmp(file)
    if tmp.suffix.lower() == ".zip":
        nii = dicom_to_nifti(tmp)
    else:
        nii = tmp

    # 2. ★★★ [중요] dt_obj 정의 부분 ★★★
    # 클라이언트가 날짜를 보냈으면 그걸 쓰고, 없으면 현재 시간을 씁니다.
    if exam_datetime:
        try:
            # ISO 포맷 등 파싱 시도
            dt_obj = datetime.strptime(exam_datetime, '%Y-%m-%d %H:%M:%S')
        except:
            try:
                # ISO 포맷 (T 포함) 파싱 시도
                dt_obj = datetime.fromisoformat(exam_datetime.replace('T', ' '))
            except:
                # 실패하면 현재 시간
                dt_obj = datetime.now()
    else:
        # 날짜 정보가 없으면 현재 시간
        dt_obj = datetime.now()

    # 3. 분석 실행
    pred = run_hippmapp3r(nii)
    left, right = split_left_right(pred)
    feats = compute_features(left, right, icv)
    feats["AGE"] = age
    feats["APOE4"] = apoe4
    feats["SEX_FEMALE"] = 1.0 if sex.upper().startswith("F") else 0.0
    
    # 4. AI 예측
    label, probs = infer(feats)
    summary = make_summary(label, probs, feats)

    # 5. DB 저장 (여기서 정의된 dt_obj를 사용합니다)
    # (1) 검사 정보(Exam) 저장 -> ID 획득
    exam_id = save_exam(patient_id, dt_obj) 
    
    # (2) 결과 정보(Result) 저장 (획득한 ID 사용)
    save_db(patient_id, file.filename, feats, label, probs, exam_id)
    
    # 6. 마스크 파일 처리
    mask_b64 = None
    import os
    if pred.exists():
        try:
            size_bytes = os.path.getsize(pred)
            print(f">>> DEBUG: pred exists at {pred} size={size_bytes} bytes")
            with open(pred, "rb") as f:
                data_bytes = f.read()
                print(f">>> DEBUG: read pred bytes: {len(data_bytes)}")
                mask_b64 = base64.b64encode(data_bytes).decode('utf-8')
                print(f">>> DEBUG: mask_base64 length: {len(mask_b64)}")
        except Exception as e:
            print(">>> ERROR reading pred file:", e)
            mask_b64 = None
    else:
        print(">>> DEBUG: pred file does NOT exist:", pred)
        mask_b64 = None

    # 7. 결과 반환
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
    
    

@app.get("/api/patients/{patient_id}/history", response_model=List[ExamHistoryItem])
def get_patient_history(patient_id: str):
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
        # 로그 남기고, JSON으로 명확히 반환 (CORS 미들 문제 완화)
        print("DB error in get_patient_history:", e)
        return JSONResponse({"status": "error", "message": "DB error: " + str(e)}, status_code=500)
    finally:
        conn.close()
        
        
@app.post("/api/get_plotly_3d")
async def get_plotly_3d(req: MaskRequest):
    try:
        print(">>> [3D 최종] 홈 버튼 고정 + 카메라 90도 회전 + 라벨/축 표시")
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

        # 데이터 확인
        if np.sum(data > 0) < 10:
            return JSONResponse({"status": "error", "message": "해마 데이터가 없습니다."}, status_code=400)

        # [1] 통합 중심점 계산 (겹침 해결)
        coords = np.argwhere(data > 0)
        # MRI 좌표 (z, y, x)
        z_mean = np.mean(coords[:, 0])
        y_mean = np.mean(coords[:, 1])
        x_mean = np.mean(coords[:, 2])
        
        # 전체 데이터의 무게 중심 (이 점을 0,0,0으로 맞춤)
        center = np.array([z_mean, y_mean, x_mean])

        traces = []
        label_positions = {}
        
        # 방향 표시를 위한 범위 계산용
        all_x, all_y, all_z = [], [], []

        def create_trace(label_id, color, name, text_label):
            if np.sum(data == label_id) < 10: return None
            m = (data == label_id).astype(np.uint8)
            try:
                verts, faces, _, _ = measure.marching_cubes(m, 0.5, step_size=1)
                
                # MRI(z,y,x) -> Plotly(x,y,z) 변환
                verts_xyz = np.vstack([verts[:, 2], verts[:, 1], verts[:, 0]]).T
                
                # 좌표 이동: (원래 좌표) - (통합 중심점)
                verts_centered = verts_xyz - np.array([center[2], center[1], center[0]])

                # 범위 수집
                all_x.extend(verts_centered[:, 0])
                all_y.extend(verts_centered[:, 1])
                all_z.extend(verts_centered[:, 2])

                # 라벨 위치: 각 해마의 개별 중심점
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
                    flatshading=True, # 선명하게
                    lighting=dict(ambient=0.7, diffuse=0.8, specular=0.2) # 밝게
                )
            except Exception as e:
                print(f"메쉬 오류 ({name}): {e}")
                return None

        # 왼쪽(1)=초록, 오른쪽(2)=빨강
        t1 = create_trace(1, '#27ae60', 'Left Hippocampus', "L")
        if t1: traces.append(t1)
        t2 = create_trace(2, '#e74c3c', 'Right Hippocampus', "R")
        if t2: traces.append(t2)

        if not traces:
            return JSONResponse({"status": "error", "message": "3D 모델 생성 실패"}, status_code=500)

        fig = go.Figure(data=traces)

        # [2] 어노테이션 (라벨 + 방향)
        annotations = []
        
        # L/R 라벨
        for txt, pos in label_positions.items():
            annotations.append(dict(
                showarrow=False,
                x=pos["x"], y=pos["y"], z=pos["z"] + 15, # 살짝 위로
                text=txt,
                font=dict(color=pos["color"], size=24, family="Arial Black"),
                xanchor="center", yanchor="bottom"
            ))

        # x, y, z 방향 표시
        padding = 20
        if all_x:
            max_x, max_y, max_z = max(all_x), max(all_y), max(all_z)
            # X축 (Left-Right)
            annotations.append(dict(showarrow=False, x=max_x + padding, y=0, z=0, text="x", font=dict(color="black", size=14)))
            # Y축 (Anterior)
            annotations.append(dict(showarrow=False, x=0, y=max_y + padding, z=0, text="y", font=dict(color="black", size=14)))
            # Z축 (Superior)
            annotations.append(dict(showarrow=False, x=0, y=0, z=max_z + padding, text="z", font=dict(color="black", size=14)))

        # [3] 레이아웃 & 카메라 설정 (핵심)
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
                
                # ★ 카메라 설정 (scene 안에 넣어야 Home 버튼 기준이 됨)
                camera=dict(
                    # eye: (1.5, 1.5, 1.5)가 기본 대각선 쿼터뷰
                    # y를 음수(-1.5)로 하면 시계방향으로 90도 회전된 위치
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
