pip install SimpleITK ants nibabel torch hdbet numpy -> 패키지 설치!!!

import os
from glob import glob
import SimpleITK as sitk
import ants
import nibabel as nib
import torch
from hdbet.predictor import hdbet_predict, get_hdbet_predictor
from multiprocessing import Pool
import logging
from pathlib import Path
import traceback
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
  
# 설정
CONFIG = {
    'INPUT_FOLDER': "원본 MRI 폴더",       
    'OUTPUT_FOLDER': "전처리 MRI 저장할 폴더",
    'MNI_TEMPLATE': "MNI 템플릿 경로",
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'KEEP_INTERMEDIATE': False,  # 중간 파일 보존 여부!
    'N_JOBS': 4  # 병렬 처리 수
}
  
class MRIPreprocessor:
    def __init__(self, config):
        self.config = config
        self.device = config['DEVICE']
        logging.info(f"Using device: {self.device}")
        
        # MNI 템플릿 경로 확인
        if not self.config['MNI_TEMPLATE'] or not os.path.exists(self.config['MNI_TEMPLATE']):
            raise FileNotFoundError("MNI 템플릿 경로를 설정하고 파일이 존재하는지 확인하세요.")
    
    def bias_correction(self, input_path, output_path):
        try:
            img = sitk.ReadImage(input_path)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            
            # 파라미터 설정
            corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
            corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
            corrector.SetWienerFilterNoise(0.01)
            
            corrected = corrector.Execute(img)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(corrected, output_path)
            logging.info(f"Bias correction completed: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Bias correction failed for {input_path}: {str(e)}")
            raise
    
    def affine_registration(self, moving_path, fixed_path, output_path):
        try:
            fixed = ants.image_read(fixed_path)
            moving = ants.image_read(moving_path)
            
            reg = ants.registration(
                fixed=fixed, 
                moving=moving, 
                type_of_transform='Affine',
                aff_metric='mattes',
                aff_sampling=32,
                syn_metric='CC',
                syn_sampling=32
            )
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ants.image_write(reg['warpedmovout'], output_path)
            logging.info(f"Registration completed: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Registration failed for {moving_path}: {str(e)}")
            raise
    
# 뇌 추출(Skull Stripping)
    def skull_strip(self, input_path, output_path, predictor):      
        try:
            brain_path = output_path.replace('.nii.gz', '_brain.nii.gz')
            mask_path = output_path.replace('.nii.gz', '_mask.nii.gz')
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            hdbet_predict(input_path, brain_path, predictor, keep_brain_mask=True)
            
            # 마스크 파일명 변경
            temp_mask = brain_path.replace('_brain.nii.gz', '_brain_mask.nii.gz')
            if os.path.exists(temp_mask):
                os.rename(temp_mask, mask_path)
            
            logging.info(f"Skull stripping completed: {brain_path}")
            return brain_path, mask_path
        except Exception as e:
            logging.error(f"Skull stripping failed for {input_path}: {str(e)}")
            raise
    
# 강도 정규화
    def intensity_normalization(self, input_path, output_path, method='z_score'):
        try:
            img = nib.load(input_path)
            data = img.get_fdata()
            
            # 뇌 영역만 고려한 정규화 (0이 아닌 값들만)
            if method == 'z_score':
                brain_mask = data > 0
                if brain_mask.sum() > 0:
                    mean_val = data[brain_mask].mean()
                    std_val = data[brain_mask].std()
                    data_norm = np.where(brain_mask, (data - mean_val) / std_val, 0)
                else:
                    data_norm = data
            elif method == 'min_max':
                brain_mask = data > 0
                if brain_mask.sum() > 0:
                    min_val = data[brain_mask].min()
                    max_val = data[brain_mask].max()
                    data_norm = np.where(brain_mask, (data - min_val) / (max_val - min_val), 0)
                else:
                    data_norm = data
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            nib.save(nib.Nifti1Image(data_norm, affine=img.affine), output_path)
            logging.info(f"Intensity normalization completed: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Intensity normalization failed for {input_path}: {str(e)}")
            raise
    # 특정 전처리 단계의 출력 파일 경로 생성
    def _get_output_path(self, subject_folder, base_name, step_name):
        step_map = {
            'corrected': '01_corrected.nii.gz',
            'registered': '02_registered.nii.gz',
            'skullstrip': '03_skullstrip.nii.gz',
            'preprocessed': 'preprocessed.nii.gz',
            'mask': 'brain_mask.nii.gz'
        }
        step_filename = step_map.get(step_name)
        if not step_filename:
            raise ValueError(f"Unknown step_name: {step_name}")
        
        file_output_folder = os.path.join(subject_folder, base_name)
        return os.path.join(file_output_folder, step_filename)
    
    # Bias Correction 단계 실행
    def _run_bias_correction(self, mri_file, output_subject_folder, base_name):
        corrected_path = self._get_output_path(output_subject_folder, base_name, 'corrected')
        self.bias_correction(mri_file, corrected_path)
        return corrected_path

    # Registration 단계 실행
    def _run_registration(self, corrected_path, output_subject_folder, base_name):
        registered_path = self._get_output_path(output_subject_folder, base_name, 'registered')
        self.affine_registration(corrected_path, self.config['MNI_TEMPLATE'], registered_path)
        return registered_path

    # Skull Stripping 단계 실행
    def _run_skull_strip(self, registered_path, output_subject_folder, base_name, predictor):
        skullstrip_path = self._get_output_path(output_subject_folder, base_name, 'skullstrip')
        brain_file, brain_mask = self.skull_strip(registered_path, skullstrip_path, predictor)
        return brain_file, brain_mask

    # Intensity Normalization 단계 실행
    def _run_normalization(self, brain_file, output_subject_folder, base_name):
        final_preprocessed_path = self._get_output_path(output_subject_folder, base_name, 'preprocessed')
        self.intensity_normalization(brain_file, final_preprocessed_path)
        return final_preprocessed_path
    
    def preprocess_subject(self, subject_path):
        try:
            predictor = get_hdbet_predictor(device=self.device)
            
            subject_id = os.path.basename(subject_path.rstrip("/"))
            output_subject_folder = os.path.join(self.config['OUTPUT_FOLDER'], subject_id)
            os.makedirs(output_subject_folder, exist_ok=True)
            
            nii_patterns = ['*.nii.gz', '*.nii']
            mri_files = []
            for pattern in nii_patterns:
                mri_files.extend(glob(os.path.join(subject_path, pattern)))
            
            if not mri_files:
                logging.warning(f"No NIfTI files found in {subject_path}")
                return
            
            processed_count = 0
            for mri_file in mri_files:
                try:
                    base_name = Path(mri_file).stem
                    if base_name.endswith('.nii'):
                        base_name = base_name[:-4]
                    
                    # 출력 파일 경로
                    file_output_folder = os.path.join(output_subject_folder, base_name)
                    final_preprocessed_path = self._get_output_path(output_subject_folder, base_name, 'preprocessed')
                    final_mask_path = self._get_output_path(output_subject_folder, base_name, 'mask')
                    
                    # 처리된 파일 건너뜀
                    if os.path.exists(final_preprocessed_path) and os.path.exists(final_mask_path):
                        logging.info(f"{subject_id}/{base_name} already processed, skipping.")
                        processed_count += 1
                        continue
                    
                    os.makedirs(file_output_folder, exist_ok=True)
                    logging.info(f"[{subject_id}] Processing {base_name}...")
                    
                    # 전처리 단계들을 별도의 메서드로 호출
                    corrected_path = self._run_bias_correction(mri_file, output_subject_folder, base_name)
                    registered_path = self._run_registration(corrected_path, output_subject_folder, base_name)
                    brain_file, brain_mask = self._run_skull_strip(registered_path, output_subject_folder, base_name, predictor)
                    self._run_normalization(brain_file, output_subject_folder, base_name)
                    
                    # 마스크 파일 최종 위치로 복사
                    if os.path.exists(brain_mask):
                        os.rename(brain_mask, final_mask_path)
                    
                    # 중간 파일 정리
                    if not self.config['KEEP_INTERMEDIATE']:
                        intermediate_files = [corrected_path, registered_path, brain_file]
                        for f in intermediate_files:
                            if os.path.exists(f):
                                os.remove(f)
                    
                    processed_count += 1
                    logging.info(f"[{subject_id}] Successfully processed {base_name}")
                    
                except Exception as e:
                    logging.error(f"Failed to process {subject_id}/{base_name}: {str(e)}")
                    logging.error(traceback.format_exc())
                    continue
            
            logging.info(f"[{subject_id}] Completed: {processed_count}/{len(mri_files)} files processed")
            
        except Exception as e:
            logging.error(f"Failed to process subject {subject_path}: {str(e)}")
            logging.error(traceback.format_exc())
    
    #전체 데이터셋 전처리
    def preprocess_all(self, use_multiprocessing=False):
        try:
            subject_folders = glob(os.path.join(self.config['INPUT_FOLDER'], "*/"))
            if not subject_folders:
                logging.warning(f"No subject folders found in {self.config['INPUT_FOLDER']}")
                return
            
            logging.info(f"Found {len(subject_folders)} subjects to process")
            
            if use_multiprocessing and len(subject_folders) > 1:
                logging.info(f"Using multiprocessing with {self.config['N_JOBS']} processes")
                with Pool(self.config['N_JOBS']) as pool:
                    pool.map(self.preprocess_subject, subject_folders)
            else:
                for subject_path in subject_folders:
                    self.preprocess_subject(subject_path)
            
            logging.info("All preprocessing completed!")
            
        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            logging.error(traceback.format_exc())
    
def main():
    try: 
        # CONFIG 값을 다시 할당할 필요 없이 바로 사용
        os.makedirs(CONFIG['OUTPUT_FOLDER'], exist_ok=True)
        
        preprocessor = MRIPreprocessor(CONFIG)
        preprocessor.preprocess_all(use_multiprocessing=False)
        
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        logging.error(traceback.format_exc())