-- DB 초기화
DROP DATABASE IF EXISTS memora_db;
CREATE DATABASE memora_db DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE memora_db;

-- 1) 환자 정보 테이블
CREATE TABLE patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,mri_results
    sex ENUM('M','F') NOT NULL,
    birth_year INT,
    birth_month TINYINT,
    birth_day TINYINT,
    height_cm INT,
    weight_kg INT,
    icv FLOAT,
    apoe4 INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 샘플 환자 데이터
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


-- 2) MRI 검사 이벤트 테이블
-- db_utils.save_exam() 이 사용하는 구조 (patient_id, exam_datetime)
CREATE TABLE mri_exams (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(100) NOT NULL,
    exam_datetime DATETIME NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_exams_patient (patient_id),
    CONSTRAINT fk_exams_patient
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


-- 3) MRI 분석 결과 테이블
-- db_utils.save_result(), fetch_exam_by_id(), fetch_patient_history() 가 사용하는 구조
CREATE TABLE mri_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(100) NOT NULL,             -- 환자 ID
    label VARCHAR(10) NOT NULL,                   -- CN or AD
    prob_cn FLOAT,
    prob_ad FLOAT,
    left_hipp_vol INT,
    right_hipp_vol INT,
    total_hipp_vol INT,
    icv FLOAT,
    age INT,
    sex VARCHAR(10),
    apoe4 INT,
    filename VARCHAR(255),                        -- 업로드된 파일명 (uploads/ 아래 파일과 매칭)
    exam_id INT NULL,                             -- mri_exams.id

    -- 마스크 메타데이터 (옵션)
    mask_size_bytes BIGINT NULL,
    mask_md5 VARCHAR(64) NULL,
    mask_sha256 VARCHAR(128) NULL,
    mask_voxel_count INT NULL,
    mask_dims VARCHAR(64) NULL,
    pred_filepath VARCHAR(255) NULL,
    pred_resampled TINYINT DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_results_patient (patient_id),
    INDEX idx_resulmri_resultsts_exam (exam_id),
    CONSTRAINT fk_mri_results_exam
        FOREIGN KEY (exam_id) REFERENCES mri_exams(id)
        ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

## 검사 이력 삭제
DELETE FROM mri_results
WHERE id > 0;

DELETE FROM mri_exams
WHERE id > 0;

SELECT patient_id, COUNT(*) as count 
FROM mri_results 
GROUP BY patient_id;

SELECT patient_id, exam_id, filename, created_at 
FROM mri_results 
ORDER BY created_at DESC;