- data/

	원본 데이터&전처리된 데이터 저장 (ex. OASIS-3 NIfTI, demographics, cls.csv 등)


- preprocessing/ 

	데이터 정리&매핑
	* [mapper.py] : OASIS-3 데이터를 읽어서 `cls.csv`로 정리


- models/

	모델 정의&학습
	* [train.py] : seg, cls 명령어 지원, 
	* [model_design.py] : 모델 구조 정의 (train.py에서 사용)


- inference/

	학습된 모델로 예측 및 해석  
	* [infer.py] : seg, cls, fuse / 특징 추출
	* [gradcam3d.py] : 시각화, XAI

- outputs/
	실행 결과 저장 (ex. volumes.csv, logits.csv, fused.csv, Grad-CAM 등)


_____________________________________________________________

실행 순서

1. 전처리
	[preprocessing/mapper.py] 실행 → [data/cls.csv] 생성  

2. 학습  
	[models/train.py] seg 실행 → 해마 추출 모델 학습

	[models/train.py] cls 실행 → AD 분류 모델 학습
	
3. 추론 
	[inference/infer.py] seg 실행 → 해마 특징 추출

	[inference/infer.py] cls 실행 → AD 확률 추론
	  

4. 병합
	[inference/infer.py] fuse 실행 → 특징이랑 확률 합쳐서 최종 결과 생성

5. 해석
	[inference/gradcam3d.py] 실행 → 결과 확인  
