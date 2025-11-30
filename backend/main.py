# backend/main.py
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import base64

from .config import UPLOAD_DIR
from .schemas import ProcessResult, PatientOut, MaskRequest, ExamHistoryItem
from .file_utils import save_file_permanent
from .mri_pipeline import parse_exam_datetime, process_mri_file, make_summary
from .db_utils import fetch_patients, fetch_exam_by_id, fetch_patient_history, update_detailed_opinion
from .plot3d_utils import generate_plotly_from_mask_b64
from .db_utils import update_detailed_opinion

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.options("/api/process_mri")
async def options_handler():
    return JSONResponse(status_code=200)


@app.get("/api/patients", response_model=list[PatientOut])
def get_patients():
    rows = fetch_patients()
    return rows


@app.post("/api/process_mri", response_model=ProcessResult)
async def process_mri(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    age: float | None = Form(None),
    apoe4: int | None = Form(None),
    sex: str | None = Form(None),
    icv: float | None = Form(None),
    exam_datetime: str | None = Form(None),
):
    uploaded_path = save_file_permanent(file)
    dt_obj = parse_exam_datetime(exam_datetime)

    result_dict = process_mri_file(
        uploaded_path=uploaded_path,
        patient_id=patient_id,
        age=age,
        apoe4=apoe4,
        sex=sex,
        icv_input=icv,
        exam_dt=dt_obj,
    )

    return JSONResponse(ProcessResult(**result_dict).dict())


@app.get("/api/exams/{exam_id}")
def get_exam_detail(exam_id: int):
    row = fetch_exam_by_id(exam_id)
    if not row:
        return JSONResponse({"status": "error", "message": "데이터 없음"}, status_code=404)

    feats = {
        "icv": row["icv"],
        "left_hipp_vol_mm3": row["left_hipp_vol"],
        "right_hipp_vol_mm3": row["right_hipp_vol"],
        "total_hipp_vol_mm3": row["total_hipp_vol"],
        "APOE4": row["apoe4"],
        "left_hipp_vol_icv_norm": None,
        "right_hipp_vol_icv_norm": None,
        "total_hipp_vol_icv_norm": None,
    }

    if row["icv"] and row["icv"] > 0:
        scale = 1000.0 / row["icv"]
        feats["left_hipp_vol_icv_norm"] = round(row["left_hipp_vol"] * scale, 3)
        feats["right_hipp_vol_icv_norm"] = round(row["right_hipp_vol"] * scale, 3)
        feats["total_hipp_vol_icv_norm"] = round(row["total_hipp_vol"] * scale, 3)
        feats["total_hipp_vol_zscore"] = row.get("total_hipp_vol_zscore")
        feats["left_hipp_vol_zscore"] = row.get("left_hipp_vol_zscore")
        feats["right_hipp_vol_zscore"] = row.get("right_hipp_vol_zscore")

    probs = {"CN": row["prob_cn"], "AD": row["prob_ad"]}
    summary = make_summary(row["label"], probs, feats)

    # 1) 원본 NIfTI 경로 (2D 뷰어용)
    file_rel = (row["filename"] or "").replace("\\", "/")
    if file_rel.startswith("uploads/"):
        file_rel = file_rel[len("uploads/"):]
    file_url = f"http://127.0.0.1:8000/uploads/{file_rel}"

    # 2) 세그 마스크를 base64로 읽어서 3D용으로 넘김
    mask_b64 = None
    pred_rel = row.get("pred_filepath")
    if pred_rel:
        pred_rel = pred_rel.replace("\\", "/")
        
        if pred_rel.startswith("uploads/"):
            pred_rel = pred_rel[len("uploads/"):]
        mask_path = Path(UPLOAD_DIR) / pred_rel
        try:
            with open(mask_path, "rb") as f:
                mask_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print("mask load error:", e)

    return {
        "status": "success",
        "data": {
            "file_url": file_url,      # ← 원본 NIfTI (2D)
            "filename": file_rel,
            "features": feats,
            "summary": summary,
            "label": row["label"],
            "probs": probs,
            "mask_base64": mask_b64,   # ← 세그 마스크 (3D용)
            "detailed_opinion": row.get("detailed_opinion") or "",
            "total_hipp_vol_zscore": row.get("total_hipp_vol_zscore"),
            "left_hipp_vol_zscore": row.get("left_hipp_vol_zscore"),
            "right_hipp_vol_zscore": row.get("right_hipp_vol_zscore"),
        },
    }


@app.get("/api/patients/{patient_id}/history", response_model=list[ExamHistoryItem])
def get_patient_history_api(patient_id: str):
    try:
        rows = fetch_patient_history(patient_id)
        history = [
            ExamHistoryItem(
                exam_id=row["exam_id"],
                exam_datetime=str(row["exam_datetime"]),
                label=row["label"],
                total_hipp_vol=row["total_hipp_vol"],
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]
        return history
    except Exception as e:
        print("DB error in get_patient_history:", e)
        return JSONResponse(
            {"status": "error", "message": "DB error: " + str(e)},
            status_code=500,
        )


@app.post("/api/get_plotly_3d")
async def get_plotly_3d(req: MaskRequest):
    try:
        fig_dict = generate_plotly_from_mask_b64(req.mask_base64)
        return JSONResponse(content=fig_dict)
    except ValueError as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    
@app.post("/api/exams/{exam_id}/opinion")
def update_opinion_api(exam_id: int, detailed_opinion: str = Form(...)):
    try:
        # DB 유틸리티 함수를 사용하여 상세 소견 업데이트
        update_detailed_opinion(exam_id, detailed_opinion)
        return JSONResponse({"status": "success", "message": "상세 소견이 성공적으로 저장되었습니다."})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"status": "error", "message": "상세 소견 저장 중 오류가 발생했습니다: " + str(e)}, 
            status_code=500
        )   
