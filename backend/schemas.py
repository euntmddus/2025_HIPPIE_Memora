# backend/schemas.py
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict


class ProcessResult(BaseModel):
    label: str
    probs: Dict[str, float]
    summary: str
    features: Dict[str, float | int | None]
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
