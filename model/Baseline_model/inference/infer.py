import os, glob, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import nibabel as nib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import measure
import json
warnings.filterwarnings('ignore')

from model_design import improved_vnet_segmentation, improved_resnet3d_classifier

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

def save_nii(p, arr, aff=None):
    if aff is None: aff = np.eye(4)
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), aff), p)

def crop_or_pad_to(shape, arr):
    target=np.array(shape); src=np.array(arr.shape)
    out=np.zeros(shape, dtype=arr.dtype)
    ssrc=np.maximum(0, (src - target)//2); esrc=ssrc + np.minimum(src, target)
    sdst=np.maximum(0, (target - src)//2); edst=sdst + np.minimum(src, target)
    out[sdst[0]:edst[0], sdst[1]:edst[1], sdst[2]:edst[2]] = arr[ssrc[0]:esrc[0], ssrc[1]:esrc[1], ssrc[2]:esrc[2]]
    return out

def test_time_augmentation_seg(model, volume):
    predictions = []
    predictions.append(model.predict(volume[None,...,None], verbose=0)[0,...,0])

    axes_to_flip = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    for axes in axes_to_flip:
        flipped_vol = np.flip(volume, axis=axes)
        pred = model.predict(flipped_vol[None,...,None], verbose=0)[0,...,0]
        pred_restored = np.flip(pred, axis=axes)
        predictions.append(pred_restored)
        
    return np.mean(predictions, axis=0)

def post_process_mask(mask_pred, threshold=0.5, min_component_size=100):
    mask_bin = (mask_pred > threshold).astype(int)
    labeled_mask, num_features = ndimage.label(mask_bin)
    if num_features == 0:
        return np.zeros_like(mask_pred, dtype=np.uint8)
    
    component_sizes = np.bincount(labeled_mask.ravel())
    
    if len(component_sizes) > 1:
        largest_component_label = component_sizes[1:].argmax() + 1
        clean_mask = (labeled_mask == largest_component_label).astype(np.uint8)
        
        if component_sizes[largest_component_label] < min_component_size:
             return np.zeros_like(mask_pred, dtype=np.uint8)
    else:
         return np.zeros_like(mask_pred, dtype=np.uint8)
        
    return ndimage.binary_fill_holes(clean_mask).astype(np.uint8)

def calculate_advanced_metrics(mask, spacing):
    metrics = {}
    voxel_count = int(np.sum(mask))
    metrics['hippo_voxels'] = voxel_count
    
    if voxel_count == 0:
        metrics['hippo_volume_mm3'] = 0.0
        metrics['surface_area_mm2'] = 0.0
        metrics['elongation'] = 1.0
        return metrics

    voxel_volume_mm3 = np.prod(spacing)
    metrics['hippo_volume_mm3'] = voxel_count * voxel_volume_mm3

    try:
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)
        metrics['surface_area_mm2'] = measure.mesh_surface_area(verts, faces)
        
        labeled_mask = measure.label(mask)
        props = measure.regionprops(labeled_mask, spacing=spacing)[0]
        metrics['solidity'] = props.solidity
        metrics['extent'] = props.extent
        
        moments = props.moments_central
        covariance = np.array([
            [moments[2, 0, 0], moments[1, 1, 0], moments[1, 0, 1]],
            [moments[1, 1, 0], moments[0, 2, 0], moments[0, 1, 1]],
            [moments[1, 0, 1], moments[0, 1, 1], moments[0, 0, 2]]
        ])
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.sqrt(eigenvalues)
        if eigenvalues[0] > 1e-6:
            metrics['elongation'] = eigenvalues[2] / eigenvalues[0]
        else:
            metrics['elongation'] = 1.0
            
    except Exception as e:
        print(f"Warning: Could not compute advanced metrics: {e}")
        if 'surface_area_mm2' not in metrics: metrics['surface_area_mm2'] = 0.0
        if 'elongation' not in metrics: metrics['elongation'] = 1.0

    return metrics

def bbox_from_mask(mask, pad=12):
    idx = np.where(mask>0.5)
    if len(idx[0])==0: 
        return (0,mask.shape[0],0,mask.shape[1],0,mask.shape[2])
    zmin,zmax = int(np.min(idx[0])), int(np.max(idx[0]))
    ymin,ymax = int(np.min(idx[1])), int(np.max(idx[1]))
    xmin,xmax = int(np.min(idx[2])), int(np.max(idx[2]))
    zmin=max(0,zmin-pad); ymin=max(0,ymin-pad); xmin=max(0,xmin-pad)
    zmax=min(mask.shape[0]-1,zmax+pad)
    ymax=min(mask.shape[1]-1,ymax+pad)
    xmax=min(mask.shape[2]-1,xmax+pad)
    return zmin,zmax+1,ymin,ymax+1,xmin,xmax+1

def extract_patch(vol, msk, size=96, ring=12):
    z0,z1,y0,y1,x0,x1 = bbox_from_mask(msk, pad=ring)
    roi = z(vol[z0:z1,y0:y1,x0:x1])
    return crop_or_pad_to((size,size,size), roi)[...,None]

def ensemble_prediction_cls(models, patch, method='average'):
    predictions, uncertainties = [], []
    
    for model in models:
        result = model.predict(patch[None,...], verbose=0)
        if isinstance(result, list) and len(result) == 2:
            pred_main, pred_unc = result
            predictions.append(pred_main[0,0])
            uncertainties.append(pred_unc[0,0])
        else:
            predictions.append(result[0,0])
            uncertainties.append(0.0)

    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)

    if method == 'average':
        final_pred = np.mean(predictions)
    elif method == 'weighted':
        weights = 1.0 / (uncertainties + 1e-6)
        weights /= np.sum(weights)
        final_pred = np.sum(predictions * weights)
    else: # median
        final_pred = np.median(predictions)
        
    epistemic_uncertainty = np.std(predictions) 
    aleatoric_uncertainty = np.mean(uncertainties)
    
    return final_pred, aleatoric_uncertainty, epistemic_uncertainty


def advanced_fusion_model(seg_features, cls_features, labels=None, method='xgboost'):
    features = []
    feature_names = []
    
    for key, values in seg_features.items():
        if key != 'id':
            features.append(values)
            feature_names.append(key)
    
    for key, values in cls_features.items():
        if key != 'id':
            features.append(values)
            feature_names.append(key)
    
    X = np.column_stack(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if labels is not None:
        y = np.array(labels)
        
        if method == 'xgboost':
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                          random_state=42, eval_metric='auc', use_label_encoder=False)
                model.fit(X_scaled, y)
                return model, scaler, feature_names
            except ImportError:
                print("XGBoost not available, falling back to Random Forest")
                method = 'rf'
        
        if method == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_scaled, y)
            return model, scaler, feature_names
        
        elif method == 'neural':
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000,
                                  random_state=42, early_stopping=True)
            model.fit(X_scaled, y)
            return model, scaler, feature_names
        
        else: 
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_scaled, y)
            return model, scaler, feature_names
    
    return None, scaler, feature_names

def visualize_results(seg_results, cls_results, fused_results, out_dir):
    plot_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(plot_dir, exist_ok=True)
    
    if 'hippo_voxels' in seg_results.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        sns.histplot(seg_results['hippo_voxels'], bins=30, ax=axes[0,0], kde=True)
        axes[0,0].set_xlabel('Hippocampus Volume (voxels)')
        axes[0,0].set_title('Hippocampus Volume Distribution')
        
        if 'prob_AD' in cls_results.columns:
            if 'hippo_voxels' not in fused_results.columns:
                 fused_results = pd.merge(fused_results, seg_results[['id', 'hippo_voxels']], on='id', how='left')

            sns.scatterplot(data=fused_results, x='hippo_voxels', y='prob_AD', hue='label' if 'label' in fused_results.columns else None, ax=axes[0,1], alpha=0.6)
            axes[0,1].set_title('Volume vs AD Probability')
            
            corr = fused_results['hippo_voxels'].corr(fused_results['prob_AD'])
            axes[0,1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                           transform=axes[0,1].transAxes, bbox=dict(boxstyle="round", facecolor='white'))
        
        sns.histplot(np.log1p(seg_results['hippo_voxels']), bins=30, ax=axes[1,0], kde=True)
        axes[1,0].set_xlabel('Log(1 + Hippocampus Volume)')
        
        if 'prob_AD' in cls_results.columns:
            sns.histplot(data=fused_results, x='prob_AD', hue='label' if 'label' in fused_results.columns else None, bins=30, ax=axes[1,1], kde=True, multiple="stack")
            axes[1,1].set_xlabel('AD Probability')
            axes[1,1].set_title('Classification Probability Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'volume_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    if 'label' in fused_results.columns and 'fused_prob_AD' in fused_results.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        y_true = fused_results['label'].values
        
        models_to_plot = []
        if 'prob_AD' in fused_results.columns:
            models_to_plot.append(('Classification Only', fused_results['prob_AD'].values))
        if 'fused_prob_AD' in fused_results.columns:
            models_to_plot.append(('Fused Model', fused_results['fused_prob_AD'].values))
        
        for name, y_scores in models_to_plot:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
            axes[0].plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
        
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0].set_title('ROC Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        for name, y_scores in models_to_plot:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            axes[1].plot(recall, precision, label=name, linewidth=2)
        
        axes[1].set_title('Precision-Recall Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Enhanced infer.py — seg/cls 추론 + 고급 퓨전")
    sub = ap.add_subparsers(dest='cmd', required=True)

    # Segmentation inference
    ps = sub.add_parser('seg')
    ps.add_argument('--images_glob', required=True)
    ps.add_argument('--models_dir', required=True, help="모델 가중치(.h5) 파일 또는 파일이 담긴 디렉토리")
    ps.add_argument('--out_dir', required=True)
    ps.add_argument('--size', type=int, default=128)
    ps.add_argument('--base', type=int, default=16)
    ps.add_argument('--th', type=float, default=0.5)
    ps.add_argument('--tta', action='store_true', default=False, help="Test Time Augmentation 활성화")
    ps.add_argument('--ensemble', action='store_true', default=False, help="앙상블 추론 활성화")
    ps.add_argument('--post_process', action='store_true', default=True, help="마스크 후처리 활성화")

    # Classification inference
    pc = sub.add_parser('cls')
    pc.add_argument('--csv', required=True, help="mapper.py로 생성된 cls.csv 파일")
    pc.add_argument('--seg_dir', required=True, help="Seg 추론으로 생성된 마스크 디렉토리")
    pc.add_argument('--models_dir', required=True, help="분류 모델 가중치(.h5) 파일 또는 디렉토리")
    pc.add_argument('--out_dir', required=True)
    pc.add_argument('--patch_size', type=int, default=96)
    pc.add_argument('--ring', type=int, default=12)
    pc.add_argument('--ensemble', action='store_true', default=False)
    pc.add_argument('--ensemble_method', choices=['average', 'weighted', 'median'], default='average')

    # Advanced fusion
    pf = sub.add_parser('fuse')
    pf.add_argument('--seg_metrics', required=True, help="seg 명령어 결과 (enhanced_volumes.csv)")
    pf.add_argument('--cls_logits', required=True, help="cls 명령어 결과 (enhanced_logits.csv)")
    pf.add_argument('--out_dir', required=True)
    pf.add_argument('--fusion_method', choices=['logistic', 'rf', 'xgboost', 'neural'], default='xgboost')
    pf.add_argument('--cv_folds', type=int, default=5, help="퓨전 모델 성능 검증용 교차 검증 폴드 수")
    pf.add_argument('--visualize', action='store_true', default=True, help="결과 시각화 리포트 생성")

    args = ap.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU setup error: {e}")

    if args.cmd == 'seg':
        print("Starting segmentation inference...")
        
        os.makedirs(args.out_dir, exist_ok=True)
        masks_out = os.path.join(args.out_dir, 'masks')
        os.makedirs(masks_out, exist_ok=True)
        
        models = []
        model_files = []
        if os.path.isdir(args.models_dir):
            model_files = glob.glob(os.path.join(args.models_dir, "*.h5")) + \
                         glob.glob(os.path.join(args.models_dir, "*", "*.h5"))
        elif os.path.isfile(args.models_dir):
             model_files = [args.models_dir]

        if not args.ensemble and len(model_files) > 1:
            print(f"Warning: Multiple models found, but --ensemble=False. Using only first model: {model_files[0]}")
            model_files = [model_files[0]]
        elif args.ensemble and not model_files:
             raise ValueError(f"No models (.h5) found in directory for ensemble: {args.models_dir}")
        
        for model_file in model_files:
            try:
                model = improved_vnet_segmentation(
                    input_shape=(args.size, args.size, args.size, 1), base=args.base
                )
                model.load_weights(model_file)
                models.append(model)
                print(f"Loaded model: {model_file}")
            except Exception as e:
                print(f"Failed to load {model_file}: {e}")
        
        if not models: raise ValueError("No models loaded successfully")
        
        paths = sorted(glob.glob(args.images_glob))
        rows = []
        
        for ip in paths:
            print(f"Processing: {Path(ip).name}")
            try:
                vol, aff = load_nii(ip)
                vol_n = crop_or_pad_to((args.size,)*3, z(vol))
                
                predictions = []
                for model in models:
                    pred = test_time_augmentation_seg(model, vol_n) if args.tta else model.predict(vol_n[None,...,None], verbose=0)[0,...,0]
                    predictions.append(pred)
                
                pred = np.mean(predictions, axis=0)
                
                mask = post_process_mask(pred, threshold=args.th) if args.post_process else (pred > args.th).astype(np.uint8)
                
                spacing = np.abs(np.diag(aff)[:3])
                metrics = calculate_advanced_metrics(mask, spacing)
                
                stem = Path(ip).stem.replace('.nii', '').replace('.gz', '')
                outp = os.path.join(masks_out, f'{stem}_hippo_mask.nii.gz')
                save_nii(outp, mask, aff)
                
                result = {'id': stem, 'path_img': ip, 'path_mask': outp, **metrics}
                rows.append(result)
                
            except Exception as e:
                print(f"Error processing {ip}: {e}")
                continue
        
        df_results = pd.DataFrame(rows)
        df_results.to_csv(os.path.join(args.out_dir, 'enhanced_volumes.csv'), index=False)
        
        summary = { 'total_processed': len(df_results), 'mean_volume_mm3': float(df_results['hippo_volume_mm3'].mean()),
            'processing_settings': { 'tta': args.tta, 'ensemble': args.ensemble, 'num_models': len(models) } }
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[DONE] Segmentation inference → {args.out_dir} ({len(df_results)} volumes processed)")

    elif args.cmd == 'cls':
        print("Starting classification inference...")
        
        os.makedirs(args.out_dir, exist_ok=True)
        df = pd.read_csv(args.csv)
        
        models = []
        model_files = []
        if os.path.isdir(args.models_dir):
            model_files = glob.glob(os.path.join(args.models_dir, "*.h5")) + glob.glob(os.path.join(args.models_dir, "*", "*.h5"))
        elif os.path.isfile(args.models_dir):
             model_files = [args.models_dir]

        if not args.ensemble and len(model_files) > 1:
            print(f"Warning: Multiple models found, but --ensemble=False. Using only first model: {model_files[0]}")
            model_files = [model_files[0]]
        elif args.ensemble and not model_files:
             raise ValueError(f"No models (.h5) found in directory for ensemble: {args.models_dir}")
        
        for model_file in model_files:
            try:
                model = improved_resnet3d_classifier(input_shape=(args.patch_size,)*3 + (1,))
                model.load_weights(model_file)
                models.append(model)
                print(f"Loaded model: {model_file}")
            except Exception as e:
                print(f"Failed to load {model_file}: {e}")
        
        if not models: raise ValueError("No models loaded successfully")

        rows = []
        for _, r in df.iterrows():
            try:
                vol,_ = load_nii(r['path'])
                stem = r.get('id', Path(r['path']).stem.replace('.nii', '').replace('.gz', ''))
                
                cand = list(Path(args.seg_dir).glob(f"{stem}*hippo_mask.nii*"))
                if not cand:
                    print(f"Warning: Mask not found for {stem}")
                    continue
                msk,_ = load_nii(str(cand[0]))
                if np.sum(msk) == 0:
                    print(f"Warning: Empty mask for {stem}")
                    continue
                
                patch = extract_patch(vol, msk, size=args.patch_size, ring=args.ring)
                
                final_prob, aleatoric_unc, epistemic_unc = ensemble_prediction_cls(models, patch, method=args.ensemble_method)
                
                rows.append({
                    'id': stem,
                    'prob_AD': float(final_prob),
                    'aleatoric_uncertainty': float(aleatoric_unc),
                    'epistemic_uncertainty': float(epistemic_unc),
                    'label': r.get('label', None)
                })
            except Exception as e:
                print(f"Error processing {r.get('id', 'unknown')}: {e}")
                continue
        
        df_results = pd.DataFrame(rows)
        df_results.to_csv(os.path.join(args.out_dir, 'enhanced_logits.csv'), index=False)
        
        summary = { 'total_processed': len(df_results), 'mean_probability': float(df_results['prob_AD'].mean()),
            'ensemble_settings': {'ensemble': args.ensemble or len(models) > 1, 'method': args.ensemble_method, 'num_models': len(models)} }
        
        if 'label' in df_results.columns and df_results['label'].notna().any():
            valid_labels_df = df_results.dropna(subset=['label'])
            y_true = valid_labels_df['label'].astype(str).str.upper().isin(['AD', '1', '1.0'])
            if len(y_true) > 0:
                summary['performance'] = {'auc': roc_auc_score(y_true, valid_labels_df['prob_AD'])}
        
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[DONE] Enhanced classification inference → {args.out_dir}")

    elif args.cmd == 'fuse':
        print("Starting advanced fusion...")
        
        os.makedirs(args.out_dir, exist_ok=True)
        
        seg = pd.read_csv(args.seg_metrics)
        cls = pd.read_csv(args.cls_logits)
        df = pd.merge(seg, cls, on='id', how='inner')
        if df.empty:
            raise ValueError("No matching samples between segmentation and classification results")
        
        feature_cols = ['prob_AD']
        if 'hippo_volume_mm3' in df.columns:
            v = df['hippo_volume_mm3'].astype(float)
            df['logV'] = np.log1p(v)
            df['zV'] = (v - v.mean())/(v.std()+1e-6)
            feature_cols.extend(['logV', 'zV'])
        
        if 'epistemic_uncertainty' in df.columns:
            feature_cols.append('epistemic_uncertainty')
        
        for col in ['surface_area_mm2', 'solidity', 'extent', 'elongation']:
             if col in df.columns:
                df[col] = df[col].fillna(0)
                if df[col].std() > 1e-6:
                    df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std()
                    feature_cols.append(f'z_{col}')
                else:
                    feature_cols.append(col)

        feature_cols = sorted(list(set(feature_cols)))
        X = df[feature_cols].values
        
        y_true_binary = None
        if 'label' in df.columns and df['label'].notna().any():
            y_true_binary = df['label'].astype(str).str.upper().isin(['AD', '1', '1.0']).astype(int).values
        
        if y_true_binary is not None:
            print(f"Training fusion model on {len(df)} samples with {len(feature_cols)} features: {feature_cols}")
            
            seg_features_dict = {c: df[c].values for c in feature_cols if c not in ['prob_AD', 'epistemic_uncertainty']}
            cls_features_dict = {c: df[c].values for c in feature_cols if c in ['prob_AD', 'epistemic_uncertainty']}
            
            model, scaler, names = advanced_fusion_model(seg_features_dict, cls_features_dict, y_true_binary, method=args.fusion_method)
            
            cv_scores = cross_val_score(model, scaler.transform(X), y_true_binary, cv=args.cv_folds, scoring='roc_auc')
            print(f"[FUSION] {args.fusion_method.upper()} CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            dump(model, os.path.join(args.out_dir, 'fusion_model.joblib'))
            dump(scaler, os.path.join(args.out_dir, 'feature_scaler.joblib'))

            X_scaled = scaler.transform(X)
            fused_probs = model.predict_proba(X_scaled)[:, 1]
            df['fused_prob_AD'] = fused_probs
            
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({'feature': names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
                importance.to_csv(os.path.join(args.out_dir, 'feature_importance.csv'), index=False)
                print("Feature importance (top 10):\n", importance.head(10))
            
            auc = roc_auc_score(y_true_binary, fused_probs)
            performance = {'auc': auc, 'cv_auc_mean': cv_scores.mean(), 'cv_auc_std': cv_scores.std()}
            print(f"[FUSION] Final Performance (on full data): AUC = {auc:.4f}")
            
            with open(os.path.join(args.out_dir, 'performance.json'), 'w') as f:
                json.dump(performance, f, indent=2)
        else:
            print("No labels found. Loading pre-trained fusion model if available...")
            try:
                model = load(os.path.join(args.out_dir, 'fusion_model.joblib'))
                scaler = load(os.path.join(args.out_dir, 'feature_scaler.joblib'))
                X_scaled = scaler.transform(X)
                df['fused_prob_AD'] = model.predict_proba(X_scaled)[:, 1]
                print("Loaded pre-trained fusion model and applied it.")
            except FileNotFoundError:
                print("No pre-trained fusion model found.")
        
        df.to_csv(os.path.join(args.out_dir, 'fused_results.csv'), index=False)
        
        if args.visualize:
            visualize_results(seg, cls, df, args.out_dir)
        
        print(f"[DONE] Advanced fusion → {args.out_dir}")

if __name__ == "__main__":
    main()