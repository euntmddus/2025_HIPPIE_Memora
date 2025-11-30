# backend/mri_pipeline.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import base64
import hashlib

import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label
from nilearn.masking import compute_brain_mask
from nilearn.image import resample_to_img

from .config import (
    HIPPMAPP3R_ENV_NAME,
    CONDA_INIT,
    ICV_FALLBACK_DEFAULT,
    xgb_model,
    FEATURE_COLS,
    UPLOAD_DIR,
)
from .file_utils import win_to_wsl, dicom_to_nifti
from .db_utils import save_exam, save_result

CN_STATS = {
    'total_hipp_vol_icv_norm': {'mean': 0.0042, 'std': 0.0007},
    'left_hipp_vol_icv_norm': {'mean': 0.0022, 'std': 0.0004},
    'right_hipp_vol_icv_norm': {'mean': 0.0021, 'std': 0.0004},
}


def calculate_zscore(norm_vol: float | None, stat_key: str) -> float | None:
    """정규화된 해마 부피를 CN 통계량과 비교하여 Z-score를 계산합니다."""
    if norm_vol is None:
        return None
        
    stats = CN_STATS.get(stat_key)
    
    if not stats or stats['std'] == 0:
        return None
    
    mean = stats['mean']
    std = stats['std']
    
    z_score = (norm_vol - mean) / std
    return round(z_score, 4)


def run_hippmapp3r(nii: Path) -> Path:

    import subprocess

    out_dir = nii.parent / "hippmapp3r"
    out_dir.mkdir(exist_ok=True)
    pred_path = out_dir / "pred.nii.gz"

    input_path = f"/data/{nii.name}"
    output_path = f"/data/{pred_path.name}"

    cmd = [
        "docker", "exec", "hippmapp3r",
        "./entrypoint.sh",
        input_path,
        output_path
    ]

    print("=== Docker CMD ===")
    print(" ".join(cmd))
    print("===================")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print("=== STDOUT ===")
    print(result.stdout)
    print("=== STDERR ===")
    print(result.stderr)

    if not pred_path.exists():
        raise RuntimeError(f"HippMapp3r Docker 실행 실패: pred 파일 없음")

    return pred_path



def largest_cc(mask: np.ndarray) -> np.ndarray:
    labeled, comp = cc_label(mask)
    if comp == 0:
        return mask
    sizes = [(labeled == i).sum() for i in range(1, comp + 1)]
    best = int(np.argmax(sizes)) + 1
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


def calculate_icv_nilearn(nifti_path: Path) -> float:
    try:
        print(f"ICV 계산 시작: {nifti_path}")
        img = nib.load(str(nifti_path))
        mask_img = compute_brain_mask(img, threshold=0.5, connected=True)
        mask_data = mask_img.get_fdata()
        voxel_count = np.sum(mask_data > 0)
        zooms = img.header.get_zooms()
        one_voxel_vol = zooms[0] * zooms[1] * zooms[2]
        icv_mm3 = voxel_count * one_voxel_vol
        print(f"ICV 계산 완료: {icv_mm3:.2f} mm³")
        return float(icv_mm3)
    except Exception as e:
        print(f"ICV 계산 실패: {e}")
        return 0.0


def compute_features(left_path: Path, right_path: Path, icv: float | None) -> dict:
    imgL = nib.load(str(left_path))
    imgR = nib.load(str(right_path))

    L = imgL.get_fdata()
    R = imgR.get_fdata()
    vox = np.prod(imgL.header.get_zooms())

    volL = float(L.sum() * vox)
    volR = float(R.sum() * vox)
    volT = volL + volR

    if volT < 100.0:
        print(f"해마 총 부피가 너무 작음 (volT={volT:.2f})")

    asym = (volL - volR) / volT if volT > 0 else 0.0

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
        feats["left_hipp_vol_icv_norm"] = round(volL / icv, 6)
        feats["right_hipp_vol_icv_norm"] = round(volR / icv, 6)
        feats["total_hipp_vol_icv_norm"] = round(volT / icv, 6)

    return feats


def build_vec(feats: dict) -> np.ndarray:
    vector = []
    for c in FEATURE_COLS:
        val = feats.get(c)
        vector.append(float(val) if val is not None else np.nan)
    return np.array(vector, dtype=np.float32).reshape(1, -1)


def infer(feats: dict):
    p = xgb_model.predict_proba(build_vec(feats))[0]
    raw = list(xgb_model.classes_)
    mapped = ["CN" if c in [0, "0"] else "AD" for c in raw]
    probs = {c: float(pp * 100) for c, pp in zip(mapped, p)}
    label = max(probs, key=probs.get)
    return label, probs


def make_summary(label: str, probs: dict, feats: dict) -> str:
    CN = round(probs["CN"])
    AD = round(probs["AD"])
    guide = "정상 범위로 예측됨." if label == "CN" else "알츠하이머 가능성이 높음."
    return (
        f"모델 예측: {label}\n"
        f"{guide}\n\n"
        f"확률: CN {CN}% · AD {AD}%\n"
        f"총 해마 부피: {feats.get('total_hipp_vol_mm3', '—')} mm³"
    )


def compute_mask_metadata(mask_path: Path, resampled: bool = False) -> dict:
    if not mask_path.exists():
        return {}

    size_bytes = mask_path.stat().st_size
    with open(mask_path, "rb") as f:
        data = f.read()

    md5 = hashlib.md5(data).hexdigest()
    sha256 = hashlib.sha256(data).hexdigest()

    img = nib.load(str(mask_path))
    data_arr = img.get_fdata()
    voxel_count = int(np.sum(data_arr > 0))
    dims_str = "x".join(str(int(d)) for d in data_arr.shape)

    try:
        #  UPLOAD_DIR을 resolve()하여 절대 경로로 변환 후 비교합니다.
        resolved_upload_dir = UPLOAD_DIR.resolve()
        rel = mask_path.resolve().relative_to(resolved_upload_dir)
        
        pred_rel = str(rel).replace("\\", "/") # 'uploads/'가 없는 형태의 전체 상대 경로가 저장됩니다.
    except ValueError:
        print("경로 계산 오류! 파일 이름만 저장됨.")
        pred_rel = mask_path.name

    return {
        "mask_size_bytes": size_bytes,
        "mask_md5": md5,
        "mask_sha256": sha256,
        "mask_voxel_count": voxel_count,
        "mask_dims": dims_str,
        "pred_filepath": pred_rel,
        "pred_resampled": 1 if resampled else 0,
    }




def parse_exam_datetime(exam_datetime: str | None) -> datetime:
    if not exam_datetime:
        return datetime.now()

    try:
        return datetime.strptime(exam_datetime, "%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            return datetime.fromisoformat(exam_datetime.replace("T", " "))
        except Exception:
            return datetime.now()


def process_mri_file(
    uploaded_path: Path,
    patient_id: str,
    age: float | None,
    apoe4: int | None,
    sex: str | None,
    icv_input: float | None,
    exam_dt: datetime,
) -> dict:
    # 1) DICOM zip 이면 NIfTI 변환
    nii = uploaded_path
    if nii.suffix.lower() == ".zip":
        nii = dicom_to_nifti(nii)

    # 2) ICV 결정 (없으면 자동 계산 + fallback)
    final_icv = icv_input
    if final_icv is None or final_icv <= 0:
        print("ICV 정보 없음 → 자동 계산")
        calculated_icv = calculate_icv_nilearn(nii)
        if calculated_icv > 0:
            final_icv = calculated_icv
        else:
            print(f"ICV 계산 실패 → fallback {ICV_FALLBACK_DEFAULT}")
            final_icv = ICV_FALLBACK_DEFAULT
    print(f"최종 적용 ICV: {final_icv}")

    # 3) 해마 세그멘테이션 + 좌/우 분리
    pred = run_hippmapp3r(nii)
    left, right = split_left_right(pred)

    # 4) 피처 + 인구학/유전자 정보
    feats = compute_features(left, right, final_icv)
    feats["AGE"] = age
    feats["APOE4"] = apoe4
    if sex is not None:
        feats["SEX_FEMALE"] = 1.0 if sex.upper().startswith("F") else 0.0
    else:
        feats["SEX_FEMALE"] = None

    # 5) 예측
    label, probs = infer(feats)
    feats['total_hipp_vol_zscore'] = calculate_zscore(
        feats.get('total_hipp_vol_icv_norm'), 'total_hipp_vol_icv_norm'
    )
    feats['left_hipp_vol_zscore'] = calculate_zscore(
        feats.get('left_hipp_vol_icv_norm'), 'left_hipp_vol_icv_norm'
    )
    feats['right_hipp_vol_zscore'] = calculate_zscore(
        feats.get('right_hipp_vol_icv_norm'), 'right_hipp_vol_icv_norm'
    )
    summary = make_summary(label, probs, feats)

    # 6) exam 저장
    exam_id = save_exam(patient_id, exam_dt)

    # 7) 마스크 리샘플링 + base64 + 메타데이터
    mask_b64 = None
    mask_meta = None
    try:
        if pred.exists():
            orig_img = nib.load(str(nii))
            pred_img = nib.load(str(pred))

            print("Mask 리샘플링 중...")
            resampled_img = resample_to_img(pred_img, orig_img, interpolation="nearest")
            resampled_data = np.round(resampled_img.get_fdata()).astype(np.uint8)

            final_mask_img = nib.Nifti1Image(resampled_data, orig_img.affine)
            final_mask_img.header.set_data_dtype(np.uint8)

            resampled_path = pred.with_name(pred.stem + "_resampled.nii.gz")
            nib.save(final_mask_img, str(resampled_path))
            print(f"리샘플링 완료: {resampled_path}")

            with open(resampled_path, "rb") as f:
                data_bytes = f.read()
                mask_b64 = base64.b64encode(data_bytes).decode("utf-8")

            mask_meta = compute_mask_metadata(resampled_path, resampled=True)
        else:
            print("pred 파일 없음")
    except Exception as e:
        print("마스크 처리 오류:", e)
        import traceback
        traceback.print_exc()
        mask_b64 = None
        mask_meta = None

    # 8) DB 저장: uploads 기준 상대 경로로 저장
    try:
        rel_path = uploaded_path.relative_to(UPLOAD_DIR)
    except ValueError:
        rel_path = uploaded_path.name

    save_result(
        patient_id,
        str(rel_path).replace("\\", "/"),
        feats,
        label,
        probs,
        exam_id,
        total_hipp_vol_zscore=feats.get('total_hipp_vol_zscore'),
        left_hipp_vol_zscore=feats.get('left_hipp_vol_zscore'),
        right_hipp_vol_zscore=feats.get('right_hipp_vol_zscore'),
        mask_meta=mask_meta,
    )

    # 9) API 응답용 dict
    return {
        "label": label,
        "probs": probs,
        "summary": summary,
        "features": feats,
        "mask_base64": mask_b64,
        "exam_datetime": exam_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "exam_id": exam_id,
        "total_hipp_vol_zscore": feats.get('total_hipp_vol_zscore'),
        "left_hipp_vol_zscore": feats.get('left_hipp_vol_zscore'),
        "right_hipp_vol_zscore": feats.get('right_hipp_vol_zscore'),
    }
