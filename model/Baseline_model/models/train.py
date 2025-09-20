import os, glob, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import nibabel as nib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from scipy import ndimage
import math

from model_design import (
    improved_vnet_segmentation, 
    improved_resnet3d_classifier,
    dice_coefficient,
    combined_dice_bce_loss,
    sensitivity,
    specificity
)

def z(x):
    x = x.astype(np.float32)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    return (x - median) / (1.4826 * mad + 1e-6)

def load_nii(p):
    try:
        img = nib.load(p)
        data = img.get_fdata(dtype=np.float32)
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, 0.0)
        return data, img.affine
    except Exception as e:
        print(f"Error loading {p}: {e}")
        raise

def crop_or_pad_to(shape, arr):
    target = np.array(shape); src = np.array(arr.shape)
    out = np.zeros(shape, dtype=arr.dtype)
    start_src = np.maximum(0, (src - target)//2); end_src = start_src + np.minimum(src, target)
    start_dst = np.maximum(0, (target - src)//2); end_dst = start_dst + np.minimum(src, target)
    out[start_dst[0]:end_dst[0], start_dst[1]:end_dst[1], start_dst[2]:end_dst[2]] = \
        arr[start_src[0]:end_src[0], start_src[1]:end_src[1], start_src[2]:end_src[2]]
    return out

def advanced_data_augmentation_3d(volume, mask, augment_prob=0.8):
    if np.random.random() > augment_prob:
        return volume, mask

    if np.random.random() > 0.5: volume, mask = np.flip(volume, axis=0).copy(), np.flip(mask, axis=0).copy()
    if np.random.random() > 0.5: volume, mask = np.flip(volume, axis=1).copy(), np.flip(mask, axis=1).copy()
    if np.random.random() > 0.5: volume, mask = np.flip(volume, axis=2).copy(), np.flip(mask, axis=2).copy()

    if np.random.random() > 0.5:
        angle = np.random.uniform(-15, 15)
        axes = tuple(np.random.choice(3, 2, replace=False))
        volume = ndimage.rotate(volume, angle, axes=axes, reshape=False, order=1)
        mask = ndimage.rotate(mask, angle, axes=axes, reshape=False, order=0)

    if np.random.random() > 0.5: volume = volume * np.random.uniform(0.8, 1.2)
    if np.random.random() > 0.5: volume = volume + np.random.normal(0, 0.05, volume.shape)
    
    return volume, (mask > 0.5).astype(np.float32)

def extract_patch(vol, msk, size=96, ring=12):
    idx = np.where(msk > 0.5)
    if len(idx[0]) == 0: 
        return np.zeros((size, size, size, 1), dtype=np.float32), False

    zmin, zmax = np.min(idx[0]), np.max(idx[0])
    ymin, ymax = np.min(idx[1]), np.max(idx[1])
    xmin, xmax = np.min(idx[2]), np.max(idx[2])
    
    z0 = max(0, zmin - ring); z1 = min(vol.shape[0], zmax + 1 + ring)
    y0 = max(0, ymin - ring); y1 = min(vol.shape[1], ymax + 1 + ring)
    x0 = max(0, xmin - ring); x1 = min(vol.shape[2], xmax + 1 + ring)
    
    roi = z(vol[z0:z1, y0:y1, x0:x1])
    patch = crop_or_pad_to((size, size, size), roi)[..., None]
    
    is_valid = np.sum(patch) > 0
    return patch, is_valid

def seg_data_generator(keys, i_dict, m_dict, size, batch_size, augment=False):
    keys = list(keys)
    
    while True:
        if augment:
            np.random.shuffle(keys)
        
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i+batch_size]
            X_batch = np.zeros((len(batch_keys), size, size, size, 1), dtype=np.float32)
            Y_batch = np.zeros((len(batch_keys), size, size, size, 1), dtype=np.float32)
            
            for j, k in enumerate(batch_keys):
                try:
                    v, _ = load_nii(i_dict[k])
                    m, _ = load_nii(m_dict[k])
                    
                    v_proc = crop_or_pad_to((size,)*3, z(v))
                    m_proc = (crop_or_pad_to((size,)*3, m) > 0.5).astype(np.float32)
                    
                    if augment:
                        v_proc, m_proc = advanced_data_augmentation_3d(v_proc, m_proc)
                    
                    X_batch[j] = v_proc[..., None]
                    Y_batch[j] = m_proc[..., None]
                except Exception as e:
                    print(f"Error loading {k} in generator: {e}")
                    continue
            yield X_batch, Y_batch

def cls_data_generator(data_pointers, seg_dir, patch_size, ring, batch_size, augment=False):
    data = list(data_pointers)
    
    while True:
        if augment:
            np.random.shuffle(data)
            
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            X_batch = np.zeros((len(batch_data), patch_size, patch_size, patch_size, 1), dtype=np.float32)
            Y_main_batch = np.zeros((len(batch_data),), dtype=np.int32)
            Y_unc_batch = np.zeros((len(batch_data),), dtype=np.float32)
            
            for j, r in enumerate(batch_data):
                try:
                    vol, _ = load_nii(r['path'])
                    mask_path = next(Path(seg_dir).glob(f"{r['id']}*hippo_mask.nii*"))
                    msk, _ = load_nii(str(mask_path))
                    
                    patch, is_valid = extract_patch(vol, msk, size=patch_size, ring=ring)
                    if not is_valid:
                        continue 

                    if augment:
                        if np.random.random() > 0.5: patch = np.flip(patch, axis=0).copy()
                        if np.random.random() > 0.5: patch = np.flip(patch, axis=1).copy()

                    X_batch[j] = patch
                    Y_main_batch[j] = r['y']
                except Exception as e:
                    print(f"Error loading {r['id']} in cls_generator: {e}")
                    continue

            yield X_batch, (Y_main_batch, Y_unc_batch)

def main():
    ap = argparse.ArgumentParser(description="Enhanced train.py for OASIS-3 project")
    sub = ap.add_subparsers(dest='cmd', required=True)

    ps = sub.add_parser('seg')
    ps.add_argument('--images_glob', required=True)
    ps.add_argument('--masks_glob', required=True)
    ps.add_argument('--out_dir', required=True)
    ps.add_argument('--size', type=int, default=128)
    ps.add_argument('--base', type=int, default=16)
    ps.add_argument('--epochs', type=int, default=200)
    ps.add_argument('--batch', type=int, default=2)
    ps.add_argument('--augment', action='store_true', default=True)
    ps.add_argument('--cross_val', type=int, default=5, help="Number of CV folds")

    pc = sub.add_parser('cls')
    pc.add_argument('--csv', required=True)
    pc.add_argument('--seg_dir', required=True)
    pc.add_argument('--out_dir', required=True)
    pc.add_argument('--patch_size', type=int, default=96)
    pc.add_argument('--ring', type=int, default=12)
    pc.add_argument('--epochs', type=int, default=150)
    pc.add_argument('--batch', type=int, default=4)
    pc.add_argument('--ensemble', type=int, default=3, help="Number of models for ensemble")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e: print(e)

    if args.cmd == 'seg':
        print("Starting segmentation training with cross-validation...")
        
        img_paths = sorted(glob.glob(args.images_glob))
        msk_paths = sorted(glob.glob(args.masks_glob))
        key_fn = lambda p: Path(p).stem.replace('.nii','').replace('.gz','').replace('_hippo_mask', '')
        i_dict = {key_fn(p):p for p in img_paths}
        m_dict = {key_fn(p):p for p in msk_paths}
        keys = np.array(sorted(set(i_dict.keys()) & set(m_dict.keys())))
        
        if len(keys) == 0:
            raise ValueError("No matching image-mask pairs found.")
            
        print(f"Found {len(keys)} matching image-mask pairs.")
        
        kfold = StratifiedKFold(n_splits=args.cross_val, shuffle=True, random_state=42)
        stratify_labels = np.random.randint(0, 2, len(keys)) 
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(keys, stratify_labels)):
            print(f"\n=== FOLD {fold+1}/{args.cross_val} ===")
            train_keys = keys[train_idx]
            val_keys = keys[val_idx]
            
            train_gen = seg_data_generator(train_keys, i_dict, m_dict, args.size, args.batch, augment=args.augment)
            val_gen = seg_data_generator(val_keys, i_dict, m_dict, args.size, args.batch, augment=False)
            
            model = improved_vnet_segmentation(input_shape=(args.size, args.size, args.size, 1), base=args.base)
            fold_dir = os.path.join(args.out_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(os.path.join(fold_dir, 'best_model.h5'), save_best_only=True, monitor='val_dice_coefficient', mode='max'),
                tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor='val_dice_coefficient', mode='max'),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=15, min_lr=1e-7),
                tf.keras.callbacks.CSVLogger(os.path.join(fold_dir, 'log.csv'))
            ]
            
            model.fit(
                train_gen, 
                validation_data=val_gen, 
                epochs=args.epochs, 
                callbacks=callbacks,
                steps_per_epoch=math.ceil(len(train_keys) / args.batch),
                validation_steps=math.ceil(len(val_keys) / args.batch)
            )
        print(f"[DONE] Segmentation training -> {args.out_dir}")

    elif args.cmd == 'cls':
        print("Starting classification training with ensembling...")
        df = pd.read_csv(args.csv)
        df['y'] = (df['label'].astype(str).str.upper() == 'AD').astype(int)
        
        valid_data_pointers = []
        for _, r in df.iterrows():
            try:
                mask_path = next(Path(args.seg_dir).glob(f"{r['id']}*hippo_mask.nii*"))
                msk, _ = load_nii(str(mask_path))
                if np.sum(msk) > 0:
                    valid_data_pointers.append(r.to_dict())
            except (FileNotFoundError, StopIteration):
                print(f"Warning: Mask not found for {r['id']}")
            except Exception as e:
                print(f"Error checking {r['id']}: {e}")

        if len(valid_data_pointers) == 0:
             raise ValueError("No valid patches found. Check --seg_dir.")
             
        print(f"Found {len(valid_data_pointers)} valid patches. Class distribution: {pd.Series([r['y'] for r in valid_data_pointers]).value_counts().to_dict()}")
        
        Y_labels = np.array([r['y'] for r in valid_data_pointers])
        class_weights = compute_class_weight('balanced', classes=np.unique(Y_labels), y=Y_labels)
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Using class weights: {class_weight_dict}")
        
        for i in range(args.ensemble):
            print(f"\n=== ENSEMBLE MODEL {i+1}/{args.ensemble} ===")
            
            train_data, val_data = train_test_split(valid_data_pointers, test_size=0.2, random_state=42 + i, stratify=Y_labels)
            
            train_gen = cls_data_generator(train_data, args.seg_dir, args.patch_size, args.ring, args.batch, augment=True)
            val_gen = cls_data_generator(val_data, args.seg_dir, args.patch_size, args.ring, args.batch, augment=False)
            
            model = improved_resnet3d_classifier(input_shape=(args.patch_size, args.patch_size, args.patch_size, 1))
            model_dir = os.path.join(args.out_dir, f'ensemble_{i}')
            os.makedirs(model_dir, exist_ok=True)
            
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'), save_best_only=True, monitor='val_main_pred_auc', mode='max'),
                tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True, monitor='val_main_pred_auc', mode='max'),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
            ]
            
            model.fit(
                train_gen, 
                validation_data=val_gen,
                epochs=args.epochs, 
                callbacks=callbacks, 
                class_weight=class_weight_dict,
                steps_per_epoch=math.ceil(len(train_data) / args.batch),
                validation_steps=math.ceil(len(val_data) / args.batch)
            )
        print(f"[DONE] Classification training -> {args.out_dir}")

if __name__ == "__main__":
    main()