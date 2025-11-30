# backend/db_utils.py
from __future__ import annotations
from typing import List, Dict, Any
import pymysql
from .config import DB_CONFIG


def get_conn():
    return pymysql.connect(**DB_CONFIG)


def save_exam(patient_id: str, exam_dt) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO mri_exams (patient_id, exam_datetime) VALUES (%s, %s)",
                (patient_id, exam_dt),
            )
            exam_id = cur.lastrowid
        conn.commit()
        return exam_id
    finally:
        conn.close()


def save_result(
    pid: str,
    filename: str,
    feats: dict,
    label: str,
    probs: dict,
    exam_id: int,
    detailed_opinion: str | None = None,
    total_hipp_vol_zscore: float | None = None,
    left_hipp_vol_zscore: float | None = None,
    right_hipp_vol_zscore: float | None = None,
    mask_meta: dict | None = None,
):
    mask_meta = mask_meta or {}
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mri_results 
                (
                    patient_id, label, prob_cn, prob_ad, 
                    left_hipp_vol, right_hipp_vol, total_hipp_vol, 
                    total_hipp_vol_zscore, left_hipp_vol_zscore, right_hipp_vol_zscore,
                    icv, age, sex, apoe4, filename, exam_id,
                    mask_size_bytes, mask_md5, mask_sha256,
                    mask_voxel_count, mask_dims, pred_filepath, pred_resampled,
                    detailed_opinion
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s)
                """,
                (
                    pid,
                    label,
                    probs["CN"],
                    probs["AD"],
                    feats["left_hipp_vol_mm3"],
                    feats["right_hipp_vol_mm3"],
                    feats["total_hipp_vol_mm3"],
                    total_hipp_vol_zscore,
                    left_hipp_vol_zscore,
                    right_hipp_vol_zscore,
                    feats["icv"],
                    feats.get("AGE"),
                    "F" if feats.get("SEX_FEMALE") else "M",
                    feats.get("APOE4"),
                    filename,
                    exam_id,
                    mask_meta.get("mask_size_bytes"),
                    mask_meta.get("mask_md5"),
                    mask_meta.get("mask_sha256"),
                    mask_meta.get("mask_voxel_count"),
                    mask_meta.get("mask_dims"),
                    mask_meta.get("pred_filepath"),
                    mask_meta.get("pred_resampled", 0),
                    detailed_opinion
                ),
            )
        conn.commit()
    finally:
        conn.close()


def fetch_patients() -> List[Dict[str, Any]]:
    conn = get_conn()
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


def fetch_exam_by_id(exam_id: int) -> Dict[str, Any] | None:
    conn = get_conn()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT * FROM mri_results WHERE exam_id = %s",
                (exam_id,),
            )
            row = cur.fetchone()
        return row
    finally:
        conn.close()


def fetch_patient_history(patient_id: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            query = """
                SELECT 
                    r.exam_id,
                    DATE_FORMAT(e.exam_datetime, '%%Y-%%m-%%d %%H:%%i:%%s') as exam_datetime,
                    r.label,
                    r.total_hipp_vol,
                    r.total_hipp_vol_zscore,
                    r.left_hipp_vol_zscore,
                    r.right_hipp_vol_zscore,
                    DATE_FORMAT(r.created_at, '%%Y-%%m-%%d %%H:%%i:%%s') as created_at
                FROM mri_results r
                JOIN mri_exams e ON r.exam_id = e.id
                WHERE r.patient_id = %s
                ORDER BY e.exam_datetime DESC
            """
            cur.execute(query, (patient_id,))
            return cur.fetchall()
    finally:
        conn.close()

def update_detailed_opinion(exam_id: int, opinion: str):
    """특정 검사(exam_id)의 상세 소견(detailed_opinion)만 업데이트합니다."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE mri_results
                SET detailed_opinion = %s
                WHERE exam_id = %s
                """,
                (opinion, exam_id),
            )
        conn.commit()
    finally:
        conn.close()
