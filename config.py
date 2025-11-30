# backend/config.py
from pathlib import Path
import xgboost as xgb
import pymysql

# DB 설정
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "1111",
    "db": "memora_db",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

# 모델 / 도구 경로
MODEL_PATH = Path("models/xgb_hippo.json")
DCM2NIIX_BIN = "dcm2niix"

# WSL + HippMapp3r
HIPPMAPP3R_ENV_NAME = "hippmapp3r"
CONDA_INIT = "source ~/miniconda3/etc/profile.d/conda.sh"

# 업로드 루트
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ICV fallback (mm3)
ICV_FALLBACK_DEFAULT = 1_500_000.0

# XGBoost 모델 (프로세스당 1회 로드)
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(MODEL_PATH)

# 분류 피처 컬럼 순서
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
