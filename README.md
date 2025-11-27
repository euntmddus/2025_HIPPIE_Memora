# 2025 HIPPIE Memora
MRI 기반 해마(hippocampus) 분할 및 알츠하이머병(AD) 보조 진단을 위한 Web·AI 통합 프로토타입  
NIfTI/DICOM 업로드 → 해마 세그멘테이션 → 해마 볼륨/ICV 계산 → XGBoost AD 분류까지 자동화 → 보조 진단 모델 구축

---

## 조직 구성도
| 역할 | 이름 | 주요 담당 |
|------|------|------------|
| **PM** | 이시연 | 프로젝트 총괄, 전체 일정·리스크 관리 |
| **PL** | 이효빈 | 팀 리딩, 세부 계획 수립 |
| **CM** | 정가영 | 문서/코드/산출물 버전 관리, 배포 통제 |
| **QA** | 임호현 | E2E 파이프라인 테스트, 모델 검증 |
| **ENG1** | 임종서 | AI 처리(세그멘테이션/ICV/Feature), ML 모델 구축 |
| **ENG2** | 운승연 | Backend(FastAPI/DB), MRI Viewer, 시스템 연동·최적화 |

---

## 협업 구조
- **PM → PL**: 일정/리스크 공유  
- **PL → ENG1/ENG2**: 개발 스프린트 계획 및 업무 전달  
- **ENG1 ↔ ENG2**: AI 처리 ↔ Web 서버 연동  
- **ENG2 → CM**: 코드/배포 버전 관리  
- **QA ↔ 전체 팀**: 검증 결과 공유 및 품질 개선  

---

## 기술 스택
- **ENG1**: Python, PyTorch/TensorFlow, WSL2, ML Pipeline  
- **ENG2**: FastAPI, MySQL, Three.js/AMI.js, Docker  
- **CM**: Git/GitHub, Notion  
- **QA**: Pipeline E2E Test 

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
- MRI 2D MPR 뷰어 + 3D 시각화  
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

---

## References
- HippMapp3r: https://github.com/mattcieslak/HippMapp3r
- ADNI: http://adni.loni.usc.edu
- OASIS-3: https://www.oasis-brains.org

---
