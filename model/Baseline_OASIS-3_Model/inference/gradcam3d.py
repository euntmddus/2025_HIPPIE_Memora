"""
explain_gradcam3d.py — 3D Grad-CAM 생성 스크립트 (오류 수정 버전)

사용법 (How to Use):
1. `train.py cls`를 통해 분류 모델을 학습시키고, 가중치 파일(.h5 또는 .ckpt)을 준비합니다.
2. `infer.py seg`를 통해 Grad-CAM을 보고 싶은 환자의 해마 마스크를 미리 생성해 둡니다.
3. 아래와 같이 터미널에서 명령어를 실행합니다.

---
사용법 예시 (Usage Example):

$ python explain_gradcam3d.py \
    --ckpt "out/cls_resnet/ensemble_0/best_model.h5" \
    --csv "data/cls.csv" \
    --seg_dir "out/seg_vnet/masks" \
    --id "sub-OASIS30001_ses-d0129_T1w" \
    --patch_size 96 \
    --out_path "out/gradcam_results/gradcam_OASIS30001.nii.gz"

---
주요 인수 설명:
--ckpt:      학습된 분류 모델의 가중치 파일 경로
             (예: .../best_model.h5 또는 .../ckpt)
--csv:       `mapper.py`로 생성한 마스터 CSV 파일 (환자 정보 탐색용)
--seg_dir:   `infer.py seg`로 생성된 해마 마스크(.nii.gz)가 저장된 디렉토리
--id:        분석할 환자의 ID. CSV 파일의 'id' 컬럼과 일치해야 함.
--patch_size: 학습 시 사용했던 패치 크기 (기본값: 96)
--ring:      학습 시 사용했던 패딩/링 크기 (기본값: 12)
--out_path:  결과로 생성될 3D Grad-CAM 히트맵 NIfTI 파일의 *저장 경로*
"""

import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf
from pathlib import Path
import pandas as pd

from model_design import improved_resnet3d_classifier

def z(x):
    x = x.astype(np.float32)
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s + 1e-6)

def load_nii(p):
    img = nib.load(p)
    return img.get_fdata(dtype=np.float32), img.affine

def bbox_from_mask(mask, pad=12):
    idx = np.where(mask > 0.5)
    if len(idx[0]) == 0:
        return (0, mask.shape[0], 0, mask.shape[1], 0, mask.shape[2])
    zmin, zmax = int(np.min(idx[0])), int(np.max(idx[0]))
    ymin, ymax = int(np.min(idx[1])), int(np.max(idx[1]))
    xmin, xmax = int(np.min(idx[2])), int(np.max(idx[2]))
    zmin = max(0, zmin - pad); ymin = max(0, ymin - pad); xmin = max(0, xmin - pad)
    zmax = min(mask.shape[0]-1, zmax + pad)
    ymax = min(mask.shape[1]-1, ymax + pad)
    xmax = min(mask.shape[2]-1, xmax + pad)
    return zmin, zmax+1, ymin, ymax+1, xmin, xmax+1

def extract_patch(vol, msk, size=96, ring=12):
    z0,z1,y0,y1,x0,x1 = bbox_from_mask(msk, pad=ring)
    roi = z(vol[z0:z1,y0:y1,x0:x1])
    target = (size,size,size)
    out = np.zeros(target, dtype=np.float32)

    src = np.array(roi.shape); tgt = np.array(target)
    ssrc = np.maximum(0, (src-tgt)//2); esrc = ssrc + np.minimum(src, tgt)
    sdst = np.maximum(0, (tgt-src)//2); edst = sdst + np.minimum(src, tgt)
    out[sdst[0]:edst[0], sdst[1]:edst[1], sdst[2]:edst[2]] = roi[ssrc[0]:esrc[0], ssrc[1]:esrc[1], ssrc[2]:esrc[2]]
    return out[...,None]

def gradcam(model, img, layer_name=None):
    if layer_name is None:
        for l in reversed(model.layers):
            if isinstance(l, tf.keras.layers.Conv3D):
                layer_name = l.name
                break
    conv_layer = model.get_layer(layer_name)

    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        inputs = tf.cast(img[None,...], tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        
        if isinstance(predictions, list):
            loss = predictions[0][:, 0]
        else:
            loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        raise ValueError("Gradient 계산에 실패했습니다. 모델 아키텍처와 layer_name을 확인하세요.")

    pooled = tf.reduce_mean(grads, axis=(0,1,2,3))
    conv_maps = conv_outputs[0].numpy()
    weights = pooled.numpy()

    cam = np.zeros(conv_maps.shape[:-1], dtype=np.float32)
    for c in range(conv_maps.shape[-1]):
        cam += weights[c] * conv_maps[...,c]
    
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-6)
    return cam

def main():
    ap = argparse.ArgumentParser(description="3D Grad-CAM 생성기")
    ap.add_argument('--ckpt', required=True, help="모델 가중치 파일 경로")
    ap.add_argument('--csv', required=True, help="마스터 CSV 파일 경로")
    ap.add_argument('--seg_dir', required=True, help="해마 마스크 디렉토리")
    ap.add_argument('--id', required=True, help="분석할 샘플 ID")
    ap.add_argument('--patch_size', type=int, default=96, help="패치 크기")
    ap.add_argument('--ring', type=int, default=12, help="패딩 크기")
    ap.add_argument('--out_path', required=True, help="결과 NIfTI 파일 저장 경로")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    row = df[df['id']==args.id].iloc[0]
    vol,_ = load_nii(row['path'])

    cand = list(Path(args.seg_dir).glob(f"{args.id}*hippo_mask.nii*"))
    if not cand:
        raise FileNotFoundError(f"마스크 파일을 찾을 수 없습니다: {args.seg_dir}/{args.id}*hippo_mask.nii*")
    msk,_ = load_nii(str(cand[0]))

    patch = extract_patch(vol, msk, size=args.patch_size, ring=args.ring)
    
    print("Loading model architecture from 'model_design.py'...")
    model = improved_resnet3d_classifier(input_shape=patch.shape)
    
    try:
        model.load_weights(args.ckpt)
        print(f"Weights loaded from: {args.ckpt}")
    except Exception as e:
        print(f"--- [오류] 가중치 로드 실패 ---")
        print(f"에러: {e}")
        print(f"모델 가중치 파일({args.ckpt})이 'improved_resnet3d_classifier' 아키텍처와 호환되는지 확인하세요.")
        raise

    print(f"[{args.id}] Grad-CAM 계산 중...")
    cam = gradcam(model, patch)
    
    nib.save(nib.Nifti1Image(cam.astype(np.float32), np.eye(4)), args.out_path)
    print(f"[DONE] Grad-CAM 저장 완료 → {args.out_path}")

if __name__ == "__main__":
    main()