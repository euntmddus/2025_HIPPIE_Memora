"""
infer.py — 추론 + 퓨전 (TensorFlow 2.10, OASIS-3 호환)
- 세그멘테이션 추론: 마스크 생성 + 부피 CSV
- 분류 추론: 패치 기반 확률
- late-fusion: 로지스틱 회귀로 결합
"""
import os, glob, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import nibabel as nib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from joblib import dump, load

def z(x):
    x = x.astype(np.float32)
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s + 1e-6)

def load_nii(p):
    img = nib.load(p)
    return img.get_fdata(dtype=np.float32), img.affine

def save_nii(p, arr, aff=None):
    if aff is None: aff = np.eye(4)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), p)

def crop_or_pad_to(shape, arr):
    target = np.array(shape); src = np.array(arr.shape)
    out = np.zeros(shape, dtype=arr.dtype)
    start_src = np.maximum(0, (src - target)//2); end_src = start_src + np.minimum(src, target)
    start_dst = np.maximum(0, (target - src)//2); end_dst = start_dst + np.minimum(src, target)
    out[start_dst[0]:end_dst[0], start_dst[1]:end_dst[1], start_dst[2]:end_dst[2]] = \
        arr[start_src[0]:end_src[0], start_src[1]:end_src[1], start_src[2]:end_src[2]]
    return out

def conv3d_block(x, f, k=3, s=1):
    x = tf.keras.layers.Conv3D(f, k, strides=s, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def unet3d(input_shape=(128,128,128,1), base=16):
    i = tf.keras.Input(shape=input_shape)
    c1 = conv3d_block(i, base); c1 = conv3d_block(c1, base); p1 = tf.keras.layers.MaxPool3D()(c1)
    c2 = conv3d_block(p1, base*2); c2 = conv3d_block(c2, base*2); p2 = tf.keras.layers.MaxPool3D()(c2)
    c3 = conv3d_block(p2, base*4); c3 = conv3d_block(c3, base*4); p3 = tf.keras.layers.MaxPool3D()(c3)
    b  = conv3d_block(p3, base*8); b  = conv3d_block(b, base*8)
    u3 = tf.keras.layers.UpSampling3D()(b);  u3 = tf.keras.layers.Concatenate()([u3, c3])
    c4 = conv3d_block(u3, base*4); c4 = conv3d_block(c4, base*4)
    u2 = tf.keras.layers.UpSampling3D()(c4); u2 = tf.keras.layers.Concatenate()([u2, c2])
    c5 = conv3d_block(u2, base*2); c5 = conv3d_block(c5, base*2)
    u1 = tf.keras.layers.UpSampling3D()(c5); u1 = tf.keras.layers.Concatenate()([u1, c1])
    c6 = conv3d_block(u1, base);  c6 = conv3d_block(c6, base)
    o  = tf.keras.layers.Conv3D(1, 1, activation='sigmoid')(c6)
    m = tf.keras.Model(i, o, name="unet3d_hippo")
    return m

def cnn3d(input_shape=(96,96,96,1)):
    i = tf.keras.Input(shape=input_shape)
    x = conv3d_block(i, 32); x = tf.keras.layers.MaxPool3D()(x)
    x = conv3d_block(x, 64); x = tf.keras.layers.MaxPool3D()(x)
    x = conv3d_block(x, 128); x = tf.keras.layers.MaxPool3D()(x)
    x = conv3d_block(x, 256); x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    o = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    m = tf.keras.Model(i, o, name="cnn3d_cls")
    return m

def bbox_from_mask(mask, pad=12):
    idx = np.where(mask>0.5)
    if len(idx[0])==0: return (0,mask.shape[0],0,mask.shape[1],0,mask.shape[2])
    zmin,zmax = int(np.min(idx[0])), int(np.max(idx[0]))
    ymin,ymax = int(np.min(idx[1])), int(np.max(idx[1]))
    xmin,xmax = int(np.min(idx[2])), int(np.max(idx[2]))
    zmin=max(0,zmin-pad); ymin=max(0,ymin-pad); xmin=max(0,xmin-pad)
    zmax=min(mask.shape[0]-1,zmax+pad); ymax=min(mask.shape[1]-1,ymax+pad); xmax=min(mask.shape[2]-1,xmax+pad)
    return zmin,zmax+1,ymin,ymax+1,xmin,xmax+1

def extract_patch(vol, msk, size=96, ring=12):
    z0,z1,y0,y1,x0,x1 = bbox_from_mask(msk, pad=ring)
    roi = z(vol[z0:z1,y0:y1,x0:x1])
    target = (size,size,size)
    out = np.zeros(target, dtype=np.float32)
    src = np.array(roi.shape); tgt = np.array(target)
    ssrc = np.maximum(0, (src-tgt)//2); esrc = ssrc+np.minimum(src,tgt)
    sdst = np.maximum(0, (tgt-src)//2); edst = sdst+np.minimum(src,tgt)
    out[sdst[0]:edst[0], sdst[1]:edst[1], sdst[2]:edst[2]] = roi[ssrc[0]:esrc[0], ssrc[1]:esrc[1], ssrc[2]:esrc[2]]
    return out[...,None]

def main():
    ap = argparse.ArgumentParser(description="infer.py — seg/cls 추론 + 퓨전 (OASIS-3)")
    sub = ap.add_subparsers(dest='cmd', required=True)

    ps = sub.add_parser('seg'); 
    ps.add_argument('--images_glob', required=True, help="예: 'D:/OASIS3_BIDS/sub-*/ses-*/anat/*T1w.nii.gz'")
    ps.add_argument('--ckpt', required=True)
    ps.add_argument('--out_dir', required=True)
    ps.add_argument('--size', type=int, default=128)
    ps.add_argument('--base', type=int, default=16)
    ps.add_argument('--th', type=float, default=0.5)

    pc = sub.add_parser('cls');
    pc.add_argument('--csv', required=True, help="OASIS-3 master/split CSV (path,label,participant_id,session,...)")
    pc.add_argument('--seg_dir', required=True)
    pc.add_argument('--ckpt', required=True)
    pc.add_argument('--out_dir', required=True)
    pc.add_argument('--patch_size', type=int, default=96)
    pc.add_argument('--ring', type=int, default=12)

    pf = sub.add_parser('fuse');
    pf.add_argument('--seg_metrics', required=True)
    pf.add_argument('--cls_logits', required=True)
    pf.add_argument('--out_dir', required=True)

    args = ap.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except: pass

    if args.cmd == 'seg':
        os.makedirs(args.out_dir, exist_ok=True)
        model = unet3d(input_shape=(args.size, args.size, args.size, 1), base=args.base)
        model.load_weights(args.ckpt)
        paths = sorted(glob.glob(args.images_glob))
        masks_out = os.path.join(args.out_dir, 'masks'); os.makedirs(masks_out, exist_ok=True)
        rows = []
        for ip in paths:
            vol, aff = load_nii(ip)
            vol_n = crop_or_pad_to((args.size,)*3, z(vol))[...,None]
            pred = model.predict(vol_n[None,...], verbose=0)[0,...,0]
            mask = (pred > args.th).astype(np.uint8)
            voxels = int(mask.sum())
            outp = os.path.join(masks_out, Path(ip).stem + '_hippo_mask.nii.gz')
            save_nii(outp, mask, aff)
            rows.append({'id': Path(ip).stem, 'path_img': ip, 'path_mask': outp, 'hippo_voxels': voxels})
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, 'volumes.csv'), index=False)
        print("[DONE] Seg inference")

    elif args.cmd == 'cls':
        os.makedirs(args.out_dir, exist_ok=True)
        df = pd.read_csv(args.csv)
        model = cnn3d(input_shape=(args.patch_size,)*3 + (1,))
        model.load_weights(args.ckpt)
        rows = []
        for _, r in df.iterrows():
            vol,_ = load_nii(r['path'])
            cand = list(Path(args.seg_dir).glob(f"{Path(r['path']).stem}*hippo_mask.nii*"))
            if not cand:
                raise FileNotFoundError(f"Mask not found for {r['path']} in {args.seg_dir}")
            msk,_ = load_nii(str(cand[0]))
            patch = extract_patch(vol, msk, size=args.patch_size, ring=args.ring)
            prob = float(model.predict(patch[None,...], verbose=0)[0,0])
            rows.append({'id': Path(r['path']).stem, 'prob_AD': prob, 'label': r.get('label', None)})
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, 'logits.csv'), index=False)
        print("[DONE] Cls inference")

    elif args.cmd == 'fuse':
        os.makedirs(args.out_dir, exist_ok=True)
        seg = pd.read_csv(args.seg_metrics)
        cls = pd.read_csv(args.cls_logits)
        df = pd.merge(seg, cls, on='id', how='inner')
        v = df['hippo_voxels'].astype(float)
        df['logV'] = np.log1p(v)
        df['zV'] = (v - v.mean())/(v.std()+1e-6)
        X = df[['prob_AD','logV','zV']].values
        y = df['label'].astype(float).values if 'label' in df.columns else None
        if y is not None:
            lr = LogisticRegression(max_iter=200)
            Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
            lr.fit(Xtr,ytr)
            p = lr.predict_proba(Xva)[:,1]
            print(f"[FUSION] AUC={roc_auc_score(yva,p):.3f}  LL={log_loss(yva,p):.3f}  Brier={brier_score_loss(yva,p):.3f}")
            dump(lr, os.path.join(args.out_dir,'fusion_model.joblib'))
            df['fused_prob_AD'] = load(os.path.join(args.out_dir,'fusion_model.joblib')).predict_proba(X)[:,1]
            df.to_csv(os.path.join(args.out_dir,'fused.csv'), index=False)
            print("[DONE] Fused saved")
        else:
            print("[WARN] labels missing — skip training fusion model.")

if __name__ == "__main__":
    main()
