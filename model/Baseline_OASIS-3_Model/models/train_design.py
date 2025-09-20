"""
train_design.py — Training script for advanced designs (OASIS-3 version)
Usage examples:
  # ResNet-style classifier on patches
  python train_design.py cls_resnet \
    --csv data/cls.csv --seg_dir out/seg/masks --out_dir out/cls_resnet \
    --patch_size 96 --ring 12 --epochs 150 --batch 2

  # ResNet+Attention classifier
  python train_design.py cls_attn \
    --csv data/cls.csv --seg_dir out/seg/masks --out_dir out/cls_attn \
    --patch_size 96 --ring 12 --epochs 150 --batch 2

  # Multitask (paired full volumes + patches and seg masks)
  python train_design.py multitask \
    --images_glob "data/images/*.nii*" --seg_dir out/seg/masks \
    --csv data/cls.csv --out_dir out/multitask \
    --size 128 --patch_size 96 --ring 12 --epochs 100 --batch 1
"""
import os, glob, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf

from model_design import resnet3d_classifier, resnet3d_with_attention, multitask_seg_cls

def z(x):
    x = x.astype(np.float32)
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s + 1e-6)

def load_nii(p):
    img = nib.load(p)
    return img.get_fdata(dtype=np.float32), img.affine

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
    ap = argparse.ArgumentParser(description="train_design.py (OASIS-3)")
    sub = ap.add_subparsers(dest='cmd', required=True)

    # ResNet classifier
    pc = sub.add_parser('cls_resnet')
    for p in [pc]:
        p.add_argument('--csv', required=True)
        p.add_argument('--seg_dir', required=True)
        p.add_argument('--out_dir', required=True)
        p.add_argument('--patch_size', type=int, default=96)
        p.add_argument('--ring', type=int, default=12)
        p.add_argument('--epochs', type=int, default=150)
        p.add_argument('--batch', type=int, default=2)

    # ResNet+Attention classifier
    pa = sub.add_parser('cls_attn')
    for p in [pa]:
        p.add_argument('--csv', required=True)
        p.add_argument('--seg_dir', required=True)
        p.add_argument('--out_dir', required=True)
        p.add_argument('--patch_size', type=int, default=96)
        p.add_argument('--ring', type=int, default=12)
        p.add_argument('--epochs', type=int, default=150)
        p.add_argument('--batch', type=int, default=2)

    # Multitask model
    pm = sub.add_parser('multitask')
    pm.add_argument('--images_glob', required=True)
    pm.add_argument('--seg_dir', required=True)
    pm.add_argument('--csv', required=True)
    pm.add_argument('--out_dir', required=True)
    pm.add_argument('--size', type=int, default=128)
    pm.add_argument('--patch_size', type=int, default=96)
    pm.add_argument('--ring', type=int, default=12)
    pm.add_argument('--epochs', type=int, default=100)
    pm.add_argument('--batch', type=int, default=1)

    args = ap.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except: pass

    os.makedirs(args.out_dir, exist_ok=True)

    if args.cmd in ['cls_resnet', 'cls_attn']:
        df = pd.read_csv(args.csv)
        assert {'id','path','label'}.issubset(df.columns)
        Xp, Y = [], []
        for _, r in df.iterrows():
            vol,_ = load_nii(r['path'])
            cand = list(Path(args.seg_dir).glob(f"{r['id']}*hippo_mask.nii*"))
            if not cand: 
                continue
            msk,_ = load_nii(str(cand[0]))
            Xp.append(extract_patch(vol, msk, size=args.patch_size, ring=args.ring))
            Y.append(int(str(r['label']).upper()=='AD'))
        Xp, Y = np.stack(Xp,0), np.array(Y)

        if args.cmd == 'cls_resnet':
            model = resnet3d_classifier(input_shape=Xp.shape[1:])
        else:
            model = resnet3d_with_attention(input_shape=Xp.shape[1:])

        ckpt = os.path.join(args.out_dir, 'ckpt')
        cbs = [
            tf.keras.callbacks.ModelCheckpoint(ckpt, save_best_only=True,
                                               save_weights_only=True, monitor='val_auc'),
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True,
                                             monitor='val_auc')
        ]

        from sklearn.model_selection import train_test_split
        Xtr,Xva,ytr,yva = train_test_split(Xp, Y, test_size=0.15,
                                           random_state=42, stratify=Y)
        hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                         epochs=args.epochs, batch_size=args.batch, callbacks=cbs)
        with open(os.path.join(args.out_dir,'train_hist.json'),'w') as f:
            json.dump({k:[float(x) for x in v] for k,v in hist.history.items()}, f)
        print("[DONE]", args.cmd, "→", ckpt)

    elif args.cmd == 'multitask':
        imgs = sorted(glob.glob(args.images_glob))
        rows = pd.read_csv(args.csv)
        id2label = {r['id']: int(str(r['label']).upper()=='AD') for _,r in rows.iterrows()}
        Xseg, Yseg, Xcls, Ycls = [], [], [], []
        for ip in imgs:
            stem = Path(ip).stem
            if stem not in id2label: 
                continue
            vol,_ = load_nii(ip)
            cand = list(Path(args.seg_dir).glob(f"{stem}*hippo_mask.nii*"))
            if not cand: 
                continue
            msk,_ = load_nii(str(cand[0]))
            v_seg = z(vol); m_seg = (msk>0.5).astype(np.float32)
            def crop_pad(vol, size):
                out = np.zeros((size,size,size), dtype=vol.dtype)
                src = np.array(vol.shape); tgt = np.array((size,size,size))
                ssrc = np.maximum(0,(src-tgt)//2); esrc = ssrc+np.minimum(src,tgt)
                sdst = np.maximum(0,(tgt-src)//2); edst = sdst+np.minimum(src,tgt)
                out[sdst[0]:edst[0], sdst[1]:edst[1], sdst[2]:edst[2]] = vol[ssrc[0]:esrc[0], ssrc[1]:esrc[1], ssrc[2]:esrc[2]]
                return out
            Xseg.append(crop_pad(v_seg, args.size)[...,None])
            Yseg.append(crop_pad(m_seg, args.size)[...,None])
            Xcls.append(extract_patch(vol, msk, size=args.patch_size, ring=args.ring))
            Ycls.append(id2label[stem])
        Xseg, Yseg = np.stack(Xseg,0), np.stack(Yseg,0)
        Xcls, Ycls = np.stack(Xcls,0), np.array(Ycls)

        model = multitask_seg_cls(input_shape_seg=Xseg.shape[1:], input_shape_cls=Xcls.shape[1:])
        ckpt = os.path.join(args.out_dir, 'ckpt')
        cbs = [
            tf.keras.callbacks.ModelCheckpoint(ckpt, save_best_only=True,
                                               save_weights_only=True, monitor='val_cls_out_auc'),
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True,
                                             monitor='val_cls_out_auc')
        ]

        from sklearn.model_selection import train_test_split
        (Xseg_tr,Xseg_va,Yseg_tr,Yseg_va,
         Xcls_tr,Xcls_va,Ycls_tr,Ycls_va) = train_test_split(Xseg, Yseg, Xcls, Ycls,
                                                             test_size=0.15,
                                                             random_state=42,
                                                             stratify=Ycls)
        hist = model.fit({'seg_input':Xseg_tr,'cls_input':Xcls_tr},
                         {'seg_out':Yseg_tr,'cls_out':Ycls_tr},
                         validation_data=({'seg_input':Xseg_va,'cls_input':Xcls_va},
                                          {'seg_out':Yseg_va,'cls_out':Ycls_va}),
                         epochs=args.epochs, batch_size=args.batch, callbacks=cbs)
        with open(os.path.join(args.out_dir,'train_hist.json'),'w') as f:
            json.dump({k:[float(x) for x in v] for k,v in hist.history.items()}, f)
        print("[DONE] multitask →", ckpt)

if __name__ == "__main__":
    main()
