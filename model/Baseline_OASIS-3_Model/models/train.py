"""
train.py — 학습 스크립트 (TensorFlow 2.10, OASIS-3 버전)
- 세그멘테이션(3D U-Net) 또는 분류(3D CNN) 학습
- mapper.py(OASIS-3)에서 생성한 cls.csv(id,path,label) 기반
"""
import os, glob, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import nibabel as nib
from sklearn.model_selection import train_test_split

def z(x):
    x = x.astype(np.float32)
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s + 1e-6)

def load_nii(p):
    img = nib.load(p)
    return img.get_fdata(dtype=np.float32), img.affine

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
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.MeanIoU(num_classes=2), 'accuracy'])
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
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['AUC','accuracy'])
    return m

def seg_dataset(images_glob, masks_glob, size=128):
    img_paths = sorted(glob.glob(images_glob)); msk_paths = sorted(glob.glob(masks_glob))
    def key(p): return Path(p).stem.replace('.nii','').replace('.gz','')
    i_dict = {key(p):p for p in img_paths}; m_dict = {key(p):p for p in msk_paths}
    keys = sorted(set(i_dict.keys()) & set(m_dict.keys()))
    X, Y = [], []
    for k in keys:
        v,_ = load_nii(i_dict[k]); m,_ = load_nii(m_dict[k])
        v = crop_or_pad_to((size,)*3, z(v))[...,None]
        m = (crop_or_pad_to((size,)*3, m)>0.5).astype(np.float32)[...,None]
        X.append(v); Y.append(m)
    return np.stack(X,0), np.stack(Y,0)

def cls_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    assert {'id','path','label'}.issubset(df.columns)
    df['y'] = (df['label'].astype(str).str.upper()=='AD').astype(int)
    return df

def main():
    ap = argparse.ArgumentParser(description="train.py — seg/cls 학습")
    sub = ap.add_subparsers(dest='cmd', required=True)

    ps = sub.add_parser('seg')
    ps.add_argument('--images_glob', required=True)
    ps.add_argument('--masks_glob', required=True)
    ps.add_argument('--out_dir', required=True)
    ps.add_argument('--size', type=int, default=128)
    ps.add_argument('--base', type=int, default=16)
    ps.add_argument('--epochs', type=int, default=200)
    ps.add_argument('--batch', type=int, default=2)

    pc = sub.add_parser('cls')
    pc.add_argument('--csv', required=True)
    pc.add_argument('--seg_dir', required=True)
    pc.add_argument('--out_dir', required=True)
    pc.add_argument('--patch_size', type=int, default=96)
    pc.add_argument('--ring', type=int, default=12)
    pc.add_argument('--epochs', type=int, default=150)
    pc.add_argument('--batch', type=int, default=2)

    args = ap.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except: pass

    if args.cmd == 'seg':
        os.makedirs(args.out_dir, exist_ok=True)
        X, Y = seg_dataset(args.images_glob, args.masks_glob, size=args.size)
        Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=0.1, random_state=42)
        model = unet3d(input_shape=X.shape[1:], base=args.base)
        ckpt = os.path.join(args.out_dir, 'ckpt')
        cbs = [
            tf.keras.callbacks.ModelCheckpoint(ckpt, save_best_only=True,
                                               save_weights_only=True, monitor='val_loss'),
            tf.keras.callbacks.EarlyStopping(patience=20,
                                             restore_best_weights=True, monitor='val_loss')
        ]
        hist = model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                         epochs=args.epochs, batch_size=args.batch, callbacks=cbs)
        with open(os.path.join(args.out_dir,'train_hist.json'),'w') as f:
            json.dump({k:[float(x) for x in v] for k,v in hist.history.items()}, f)
        print("[DONE] Seg training →", ckpt)

    elif args.cmd == 'cls':
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

        os.makedirs(args.out_dir, exist_ok=True)
        df = cls_dataframe(args.csv)
        Xp, Y = [], []
        for _, r in df.iterrows():
            vol,_ = load_nii(r['path'])
            cand = list(Path(args.seg_dir).glob(f"{r['id']}*hippo_mask.nii*"))
            if not cand:
                raise FileNotFoundError(f"Mask not found for id {r['id']} in {args.seg_dir}")
            msk,_ = load_nii(str(cand[0]))
            patch = extract_patch(vol, msk, size=args.patch_size, ring=args.ring)
            Xp.append(patch); Y.append(r['y'])
        Xp = np.stack(Xp,0); Y = np.array(Y)
        Xtr, Xva, ytr, yva = train_test_split(Xp, Y, test_size=0.15,
                                              random_state=42, stratify=Y)
        model = cnn3d(input_shape=Xp.shape[1:])
        ckpt = os.path.join(args.out_dir, 'ckpt')
        cbs = [
            tf.keras.callbacks.ModelCheckpoint(ckpt, save_best_only=True,
                                               save_weights_only=True, monitor='val_auc'),
            tf.keras.callbacks.EarlyStopping(patience=15,
                                             restore_best_weights=True, monitor='val_auc')
        ]
        hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                         epochs=args.epochs, batch_size=args.batch, callbacks=cbs)
        with open(os.path.join(args.out_dir,'train_hist.json'),'w') as f:
            json.dump({k:[float(x) for x in v] for k,v in hist.history.items()}, f)
        print("[DONE] Cls training →", ckpt)

if __name__ == "__main__":
    main()
