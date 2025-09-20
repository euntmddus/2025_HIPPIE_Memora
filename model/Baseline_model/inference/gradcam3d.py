"""
explain_gradcam3d.py 

--ckpt:      학습된 분류 모델의 가중치 파일 경로 (예: .../best_model.h5)
--csv:       `mapper.py`로 생성한 마스터 CSV 파일
--seg_dir:   `infer.py seg`로 생성된 해마 마스크 디렉토리
--id:        분석할 환자의 ID (CSV의 'id'와 일치)
--patch_size: 학습 시 사용했던 패치 크기
--ring:      학습 시 사용했던 패딩/링 크기
--method:    사용할 분석 기법 ('gradcam', 'guided', 'integrated', 'all')
--out_path:  파일 저장 경로 
--visualize:   3-Panel 뷰 시각화 PNG 파일을 저장
--uncertainty: MC Dropout을 사용한 불확실성 분석을 수행합니다.
--multi_layer: 마지막 3개의 Conv 레이어를 모두 분석합니다.
"""

import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
from typing import List, Tuple, Optional, Dict, Any
import warnings
from scipy import ndimage 

from model_design import improved_resnet3d_classifier

warnings.filterwarnings('ignore')

def z(x):
    """Robust z-score normalization"""
    x = x.astype(np.float32)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    return (x - median) / (1.4826 * mad + 1e-6)

def load_nii(p):
    """Enhanced NIfTI loading"""
    try:
        img = nib.load(p)
        data = img.get_fdata(dtype=np.float32)
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, 0.0)
        return data, img.affine
    except Exception as e:
        print(f"Error loading {p}: {e}")
        raise

def bbox_from_mask(mask, pad=12):
    """Enhanced bounding box extraction"""
    idx = np.where(mask > 0.5)
    if len(idx[0]) == 0:
        return (0, mask.shape[0], 0, mask.shape[1], 0, mask.shape[2])
    
    zmin, zmax = int(np.min(idx[0])), int(np.max(idx[0]))
    ymin, ymax = int(np.min(idx[1])), int(np.max(idx[1]))
    xmin, xmax = int(np.min(idx[2])), int(np.max(idx[2]))
    
    zmin = max(0, zmin - pad)
    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    zmax = min(mask.shape[0]-1, zmax + pad)
    ymax = min(mask.shape[1]-1, ymax + pad)
    xmax = min(mask.shape[2]-1, xmax + pad)
    
    return zmin, zmax+1, ymin, ymax+1, xmin, xmax+1

def extract_patch(vol, msk, size=96, ring=12):
    """Enhanced patch extraction with quality checks"""
    z0, z1, y0, y1, x0, x1 = bbox_from_mask(msk, pad=ring)
    roi = z(vol[z0:z1, y0:y1, x0:x1])
    
    target = (size, size, size)
    out = np.zeros(target, dtype=np.float32)
    
    src = np.array(roi.shape)
    tgt = np.array(target)
    ssrc = np.maximum(0, (src-tgt)//2)
    esrc = ssrc + np.minimum(src, tgt)
    sdst = np.maximum(0, (tgt-src)//2)
    edst = sdst + np.minimum(src, tgt)
    
    out[sdst[0]:edst[0], sdst[1]:edst[1], sdst[2]:edst[2]] = \
        roi[ssrc[0]:esrc[0], ssrc[1]:esrc[1], ssrc[2]:esrc[2]]
    
    return out[..., None]

def enhanced_gradcam(model, img, layer_name=None, class_idx=0):
    """Enhanced Grad-CAM with better gradient computation"""
    if layer_name is None:
        conv_layers = [l for l in reversed(model.layers) 
                       if isinstance(l, tf.keras.layers.Conv3D)]
        if not conv_layers:
            raise ValueError("No Conv3D layers found in model")
        layer_name = conv_layers[0].name
        print(f"Using layer: {layer_name}")
    
    try:
        conv_layer = model.get_layer(layer_name)
    except ValueError:
        print(f"Layer {layer_name} not found. Available layers:")
        for l in model.layers:
            print(f"  {l.name}")
        raise
    
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        inputs = tf.cast(img[None, ...], tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        
        if isinstance(predictions, list):
            loss = predictions[0][:, class_idx]  # Main prediction
        else:
            loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        raise ValueError("Gradients are None. Check model architecture and layer name.")
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
    conv_maps = conv_outputs[0].numpy()
    weights = pooled_grads.numpy()
    
    cam = np.zeros(conv_maps.shape[:-1], dtype=np.float32)
    for c in range(conv_maps.shape[-1]):
        cam += weights[c] * conv_maps[..., c]
    
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    
    return cam, weights

def guided_gradcam(model, img, layer_name=None, class_idx=0):
    """Guided Grad-CAM implementation"""
    cam, weights = enhanced_gradcam(model, img, layer_name, class_idx)
    
    with tf.GradientTape() as tape:
        inputs = tf.cast(img[None, ...], tf.float32)
        tape.watch(inputs)
        
        if hasattr(model, 'output') and isinstance(model.output, list):
            predictions = model(inputs)[0]
        else:
            predictions = model(inputs)
        
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, inputs)
    
    if grads is not None:
        guided_grads = grads[0].numpy()
        
        cam_resized = ndimage.zoom(cam, 
                                   (img.shape[0]/cam.shape[0],
                                    img.shape[1]/cam.shape[1], 
                                    img.shape[2]/cam.shape[2]))
        
        guided_gradcam = guided_grads[..., 0] * cam_resized
        return guided_gradcam
    
    return cam

def integrated_gradients(model, img, baseline=None, steps=50, class_idx=0):
    """Integrated Gradients implementation"""
    if baseline is None:
        baseline = np.zeros_like(img)
    
    alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)
    gradients = []
    
    for alpha in alphas:
        interpolated = baseline + alpha * (img - baseline)
        with tf.GradientTape() as tape:
            inputs = tf.cast(interpolated[None, ...], tf.float32)
            tape.watch(inputs)
            
            if hasattr(model, 'output') and isinstance(model.output, list):
                predictions = model(inputs)[0]
            else:
                predictions = model(inputs)
            
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, inputs)
        if grads is not None:
            gradients.append(grads[0])
    
    if gradients:
        grads_tensor = tf.convert_to_tensor(gradients)
        avg_grads = tf.reduce_mean(grads_tensor, axis=0)
        integrated_grads = (img - baseline) * avg_grads.numpy()[..., 0]
        return integrated_grads
    
    return np.zeros_like(img[..., 0])

def analyze_multiple_layers(model, img, class_idx=0, top_k=3):
    """Analyze multiple convolutional layers"""
    conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv3D)]
    
    results = {}
    for layer in conv_layers[-top_k:]:  # Take last k layers
        try:
            cam, weights = enhanced_gradcam(model, img, layer.name, class_idx)
            results[layer.name] = {
                'cam': cam,
                'weights': weights,
                'layer_depth': len([l for l in model.layers[:model.layers.index(layer)] 
                                   if isinstance(l, tf.keras.layers.Conv3D)])
            }
        except Exception as e:
            print(f"Error analyzing layer {layer.name}: {e}")
            continue
    
    return results

def create_3d_visualizations(cam, original_img, output_path, method_name="GradCAM"):
    """Create comprehensive 3D visualizations"""

    if cam.shape != original_img.shape[:3]:
        cam_resized = ndimage.zoom(cam, 
                                   (original_img.shape[0]/cam.shape[0],
                                    original_img.shape[1]/cam.shape[1], 
                                    original_img.shape[2]/cam.shape[2]))
    else:
        cam_resized = cam
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'3D {method_name} Analysis', fontsize=16)
    
    colors = ['black', 'purple', 'blue', 'green', 'yellow', 'orange', 'red']
    cmap = LinearSegmentedColormap.from_list('gradcam', colors)
    
    views = [
        (0, 'Axial', [original_img.shape[0]//4, original_img.shape[0]//2, 3*original_img.shape[0]//4]),
        (1, 'Sagittal', [original_img.shape[1]//4, original_img.shape[1]//2, 3*original_img.shape[1]//4]),
        (2, 'Coronal', [original_img.shape[2]//4, original_img.shape[2]//2, 3*original_img.shape[2]//4])
    ]
    
    for row, (axis, name, slices) in enumerate(views):
        for col, slice_idx in enumerate(slices):
            ax = axes[row, col]
            if axis == 0:
                img_slice = original_img[slice_idx, :, :, 0]
                cam_slice = cam_resized[slice_idx, :, :]
            elif axis == 1:
                img_slice = original_img[:, slice_idx, :, 0]
                cam_slice = cam_resized[:, slice_idx, :]
            else:
                img_slice = original_img[:, :, slice_idx, 0]
                cam_slice = cam_resized[:, :, slice_idx]
                
            ax.imshow(img_slice, cmap='gray', alpha=0.7)
            im = ax.imshow(cam_slice, cmap=cmap, alpha=0.5, vmin=0.0, vmax=1.0)
            ax.set_title(f'{name} Slice {slice_idx}')
            ax.axis('off')
    
    ax = axes[0, 3]
    ax.hist(cam_resized.flatten(), bins=50, range=(0.01, 1.0), alpha=0.7)
    ax.set_title('Activation Distribution ( > 0.01)')
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Frequency')
    
    ax = axes[1, 3]
    ax.text(0.1, 0.8, f'Max Activation: {cam_resized.max():.3f}', transform=ax.transAxes)
    ax.text(0.1, 0.6, f'Mean Activation: {cam_resized.mean():.3f}', transform=ax.transAxes)
    ax.text(0.1, 0.4, f'Std Activation: {cam_resized.std():.3f}', transform=ax.transAxes)
    ax.text(0.1, 0.2, f'Active Voxels (> 0.1): {(cam_resized > 0.1).sum()}', transform=ax.transAxes)
    ax.set_title('Statistics')
    ax.axis('off')
    
    ax = axes[2, 3]
    ax.axis('off')
    
    fig.colorbar(im, ax=axes, shrink=0.6, aspect=30)
    plt.tight_layout()
    viz_path = output_path.replace('.nii.gz', '_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

def uncertainty_analysis(model, img, n_samples=20):
    """Monte Carlo dropout for uncertainty estimation"""
    
    predictions = []
    for _ in range(n_samples):
        pred = model(img[None, ...], training=True) 
        if isinstance(pred, list):
            predictions.append(pred[0][0, 0].numpy())
        else:
            predictions.append(pred[0, 0].numpy())
    
    predictions = np.array(predictions)
    
    return {
        'mean_prediction': predictions.mean(),
        'prediction_std': predictions.std(),
        'prediction_variance': predictions.var(),
        'epistemic_uncertainty': predictions.std(),
        'predictions': predictions
    }

def main():
    ap = argparse.ArgumentParser(description="Enhanced 3D Grad-CAM Analysis")
    ap.add_argument('--ckpt', required=True, help="Model checkpoint path (weights only)")
    ap.add_argument('--csv', required=True, help="Dataset CSV file")
    ap.add_argument('--seg_dir', required=True, help="Segmentation masks directory")
    ap.add_argument('--id', required=True, help="Sample ID to analyze")
    ap.add_argument('--patch_size', type=int, default=96, help="Patch size")
    ap.add_argument('--ring', type=int, default=12, help="Padding around hippocampus")
    ap.add_argument('--method', choices=['gradcam', 'guided', 'integrated', 'all'], 
                   default='gradcam', help="Attribution method")
    ap.add_argument('--layer', default=None, help="Target layer name (optional)")
    ap.add_argument('--out_path', required=True, help="Output NIfTI path (base name)")
    ap.add_argument('--visualize', action='store_true', help="Create visualizations")
    ap.add_argument('--uncertainty', action='store_true', help="Perform uncertainty analysis")
    ap.add_argument('--multi_layer', action='store_true', help="Analyze multiple layers")
    ap.add_argument('--class_idx', type=int, default=0, help="Class index for analysis")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    row = df[df['id'] == args.id]
    
    if row.empty:
        raise ValueError(f"Sample {args.id} not found in CSV")
    
    row = row.iloc[0]
    vol, vol_affine = load_nii(row['path'])

    mask_candidates = list(Path(args.seg_dir).glob(f"{args.id}*hippo_mask.nii*"))
    if not mask_candidates:
        raise FileNotFoundError(f"Mask not found for {args.id}")
    
    msk, msk_affine = load_nii(str(mask_candidates[0]))

    patch = extract_patch(vol, msk, size=args.patch_size, ring=args.ring)
    
    print("Loading model architecture from model_design.py...")
    model = improved_resnet3d_classifier(input_shape=patch.shape)
    
    try:
        model.load_weights(args.ckpt)
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Ensure the checkpoint file corresponds to the 'improved_resnet3d_classifier' architecture.")
        raise
    
    print(f"Model loaded with weights from: {args.ckpt}")
    print(f"Patch shape: {patch.shape}")

    prediction = model.predict(patch[None, ...], verbose=0)
    if isinstance(prediction, list):
        pred_prob = prediction[0][0, 0]
        uncertainty = prediction[1][0, 0] if len(prediction) > 1 else 0.0
    else:
        pred_prob = prediction[0, 0]
        uncertainty = 0.0
    
    print(f"Prediction: {pred_prob:.4f} (Aleatoric Uncertainty: {uncertainty:.4f})")
    
    results = {
        'sample_id': args.id,
        'prediction_probability': float(pred_prob),
        'model_aleatoric_uncertainty': float(uncertainty),
        'patch_shape': list(patch.shape),
        'method': args.method
    }

    if args.method == 'gradcam' or args.method == 'all':
        print("Computing Grad-CAM...")
        cam, weights = enhanced_gradcam(model, patch, args.layer, args.class_idx)
        
        gradcam_path = args.out_path.replace('.nii.gz', '_gradcam.nii.gz')
        nib.save(nib.Nifti1Image(cam.astype(np.float32), np.eye(4)), gradcam_path)
        results['gradcam_path'] = gradcam_path
        
        if args.visualize:
            viz_path = create_3d_visualizations(cam, patch, gradcam_path, "GradCAM")
            results['gradcam_visualization'] = viz_path

    if args.method == 'guided' or args.method == 'all':
        print("Computing Guided Grad-CAM...")
        guided_cam = guided_gradcam(model, patch, args.layer, args.class_idx)
        
        guided_path = args.out_path.replace('.nii.gz', '_guided.nii.gz')
        nib.save(nib.Nifti1Image(guided_cam.astype(np.float32), np.eye(4)), guided_path)
        results['guided_path'] = guided_path
        
        if args.visualize:
            viz_path = create_3d_visualizations(guided_cam, patch, guided_path, "Guided GradCAM")
            results['guided_visualization'] = viz_path

    if args.method == 'integrated' or args.method == 'all':
        print("Computing Integrated Gradients...")
        integrated_grads = integrated_gradients(model, patch, class_idx=args.class_idx)
        
        integrated_path = args.out_path.replace('.nii.gz', '_integrated.nii.gz')
        nib.save(nib.Nifti1Image(integrated_grads.astype(np.float32), np.eye(4)), integrated_path)
        results['integrated_path'] = integrated_path
        
        if args.visualize:
            viz_path = create_3d_visualizations(integrated_grads, patch, integrated_path, "Integrated Gradients")
            results['integrated_visualization'] = viz_path

    if args.multi_layer:
        print("Analyzing multiple layers...")
        layer_results = analyze_multiple_layers(model, patch, args.class_idx)
        
        for layer_name, layer_data in layer_results.items():
            layer_path = args.out_path.replace('.nii.gz', f'_layer_{layer_name}.nii.gz')
            nib.save(nib.Nifti1Image(layer_data['cam'].astype(np.float32), np.eye(4)), layer_path)
            
            if args.visualize:
                create_3d_visualizations(
                    layer_data['cam'], patch, layer_path, f"GradCAM Layer {layer_name}"
                )
        
        results['layer_analysis'] = {k: {'weights_stats': {
            'mean': float(v['weights'].mean()),
            'std': float(v['weights'].std()),
            'max': float(v['weights'].max()),
            'min': float(v['weights'].min())
        }} for k, v in layer_results.items()}

    if args.uncertainty:
        print("Performing uncertainty analysis (MC Dropout)...")
        uncertainty_results = uncertainty_analysis(model, patch)
        uncertainty_results['predictions'] = uncertainty_results['predictions'].tolist()
        results['uncertainty_analysis'] = uncertainty_results
        
        print(f"Prediction Mean: {uncertainty_results['mean_prediction']:.4f}")
        print(f"Prediction Std (Epistemic Uncertainty): {uncertainty_results['epistemic_uncertainty']:.4f}")

    results_path = args.out_path.replace('.nii.gz', '_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"[DONE] Enhanced analysis completed")
    print(f"Results JSON saved to: {results_path}")
    if args.method == 'gradcam' or args.method == 'all':
        print(f"Grad-CAM NIfTI saved to: {gradcam_path}")

if __name__ == "__main__":
    main()