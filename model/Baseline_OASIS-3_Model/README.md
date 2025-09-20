OASIS-3_Model 프로젝트

폴더 구조
- data/
  원본 데이터와 전처리된 데이터를 넣는 곳  
  (예: OASIS-3 NIfTI, demographics, cls.csv 등)

- preprocessing/ 
  데이터 정리/매핑 코드가 들어있는 폴더  
  - `mapper.py` : OASIS-3 데이터를 읽어서 `cls.csv`로 정리

- models/
  모델을 정의하고 학습하는 코드가 들어있는 폴더  
  - `train.py` : 기본 모델 학습 (U-Net, CNN3D)  
  - `train_design.py` : 고급 모델 학습 (ResNet, Attention, Multitask)  
  - `model_design.py` : 모델 구조 정의 (다른 코드에서 불러씀)

- inference/
  학습된 모델을 가지고 예측/해석하는 코드 폴더  
  - `infer.py` : 세그멘테이션/분류 실행 + fusion 결과 생성  
  - `explain_gradcam3d.py` : Grad-CAM 시각화 (어디를 보고 예측했는지 확인)

- outputs/
  실행 결과가 저장되는 폴더  
  (예: volumes.csv, logits.csv, fused.csv, Grad-CAM 이미지)

--------------------------------------------------------------------------------------------

실행 순서
1. 전처리
   `preprocessing/mapper.py` 실행 → `cls.csv` 생성  

2. 학습  
   - 기본 학습: `models/train.py`  
   - 고급 학습: `models/train_design.py`  

3. 추론 
   `inference/infer.py` 실행 → 결과 파일 생성  

4. 시각화
   `inference/explain_gradcam3d.py` 실행 → Grad-CAM 결과 확인  
