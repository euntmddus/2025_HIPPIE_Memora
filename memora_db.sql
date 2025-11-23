-- Database initialization and sample data for memora_db
-- 1) Remove existing database if present to avoid Error 1007
DROP DATABASE IF EXISTS memora_db;

-- 2) Create fresh database with utf8mb4 charset
CREATE DATABASE memora_db DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;

-- 3) Select the database to operate on to avoid Error 1046
USE memora_db;

-- 4) Create patients table (unique patient_id)
CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(100) NOT NULL UNIQUE, -- 외부에서 참조할 고유 환자 식별자
    name VARCHAR(100) NOT NULL,
    sex ENUM('M','F') NOT NULL,
    birth_year INT,
    birth_month TINYINT,
    birth_day TINYINT,
    height_cm INT,
    weight_kg INT,
    icv FLOAT NULL COMMENT 'ICV (mm3)',
    apoe4 INT NULL COMMENT 'APOE4 유전자형 (0, 1, 2)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 5) Insert sample patients (ensure unique patient_id values)
INSERT IGNORE INTO patients (patient_id, name, sex, birth_year, birth_month, birth_day, height_cm, weight_kg, apoe4)
VALUES
('P0001', '이은필', 'F', 2004, 5, 10, 165, 60, NULL),
('P0002', '문형서', 'M', 2003, 11, 2, 178, 72, 1),
('P0003', '김하늘', 'F', 1974, 1, 2, 160, 62, 2),
('P0004', '박지훈', 'M', 1983, 11, 3, 181, 80, 0),
('P0005', '최민서', 'F', 1964, 9, 2, 180, 55, 2),
('P0006', '정우진', 'M', 1990, 2, 2, 171, 56, 1),
('P0007', '한예린', 'F', 2011, 4, 7, 157, 90, 0),
('P0008', '오준석', 'M', 1953, 10, 21, 188, 72, NULL);

-- 6) Create mri_exams table to record individual exam events
CREATE TABLE IF NOT EXISTS mri_exams (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(100) NOT NULL,
    exam_datetime DATETIME NOT NULL,
    modality VARCHAR(50) DEFAULT 'MRI',
    series_description VARCHAR(255),
    note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_exams_patient (patient_id),
    CONSTRAINT fk_exams_patient 
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 7) Create mri_results table for analysis results (linkable to mri_exams)
CREATE TABLE IF NOT EXISTS mri_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(100) NOT NULL,
    label VARCHAR(10) NOT NULL,             -- 예: CN, AD
    prob_cn FLOAT,
    prob_ad FLOAT,
    left_hipp_vol INT,
    right_hipp_vol INT,
    total_hipp_vol INT,
    icv FLOAT,                              -- ICV는 실수형으로 저장
    age INT,
    sex VARCHAR(10),
    apoe4 INT,
    filename VARCHAR(255),
    exam_id INT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_results_patient (patient_id),
    INDEX idx_results_exam (exam_id),
    CONSTRAINT fk_mri_results_exam
        FOREIGN KEY (exam_id) REFERENCES mri_exams(id)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 8) Optional: small sanity-check helper view (example)
-- CREATE OR REPLACE VIEW v_patient_latest AS
-- SELECT p.patient_id, p.name, r.label, r.prob_cn, r.prob_ad, r.created_at
-- FROM patients p
-- LEFT JOIN mri_results r ON r.patient_id = p.patient_id
-- ORDER BY r.created_at DESC;

-- ------------------------------------------------------------------
-- Migration: add mask metadata columns to mri_results (db_migration_add_mask_meta.sql)
-- Add columns that store mask file info and resampling flag
-- ------------------------------------------------------------------
ALTER TABLE mri_results
  ADD COLUMN IF NOT EXISTS mask_size_bytes BIGINT NULL,
  ADD COLUMN IF NOT EXISTS mask_md5 VARCHAR(64) NULL,
  ADD COLUMN IF NOT EXISTS mask_sha256 VARCHAR(128) NULL,
  ADD COLUMN IF NOT EXISTS mask_voxel_count INT NULL,
  ADD COLUMN IF NOT EXISTS mask_dims VARCHAR(64) NULL,
  ADD COLUMN IF NOT EXISTS pred_filepath VARCHAR(255) NULL,
  ADD COLUMN IF NOT EXISTS pred_resampled TINYINT DEFAULT 0;