# 2025 HIPPIE Memora
MRI 기반 해마(hippocampus) 분할 및 알츠하이머병(AD) 보조 진단을 위한 Web·AI 통합 프로토타입  
NIfTI/DICOM 업로드 → 해마 세그멘테이션 → 해마 볼륨/ICV 계산 → XGBoost AD 분류까지 자동화 → 보조 진단 모델 구축

---

## System Architecture
- **Interface Layer**: Web + FastAPI  
- **AI Layer**: HippMapp3r segmentation(WSL2), ICV 계산, hippocampal feature extraction, XGBoost inference  
- **Service Layer**: MRI processing, DICOM→NIfTI 변환, feature 계산  
- **DB Layer**: MySQL (환자 정보 / 검사 기록 / 분석 결과 저장)  
- **Dataset**: ADNI(T1 baseline) 학습, OASIS-3 외부 데이터 테스트  

---

## Features
- DICOM → NIfTI 자동 변환  
- HippMapp3r 기반 좌/우 해마 mask 생성  
- Hippocampal volume & asymmetry 자동 계산  
- ICV(auto) 계산 및 normalization  
- XGBoost 기반 CN vs AD 분류  
- MRI 2D MPR 뷰어 + 3D Mesh 시각화  
- 검사 기록 및 결과 MySQL 저장  

---

## Machine Learning Model
- **Input features**
  - Left/Right/Total hippocampal volume  
  - Asymmetry index  
  - ICV-normalized volumes  
  - Age, Sex, APOE4  
- **Model**: XGBoost  
- **Performance**: AUC ~0.90 / Accuracy ~0.85 (ADNI baseline)  
- **External Test**: OASIS-3 raw MRI 기반 end-to-end 평가 지원  
