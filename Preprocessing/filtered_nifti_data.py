import os
import shutil
import pandas as pd


# 1. 인덱스 파일 경로 (CSV 파일)
INDEX_FILE_PATH = 'D:/HIPPIE/ADNI_Project/metadata/train_index.csv' 

# 2. 원본 파일이 있는 폴더
TARGET_DIR = 'D:/HIPPIE/ADNI_Project/raw_nifti'

# 3. 새로 파일을 복사할 폴더
NEW_TARGET_DIR = 'D:/HIPPIE/ADNI_Project/filtered_nifti_data'

# CSV 파일 내용 중 'raw_nifti/' 제거
BASE_PATH_TO_REMOVE = 'raw_nifti/'
# -----------------


def copy_matching_files_flat_cleaned(index_path, target_dir, new_target_dir, path_to_remove):
    """
    CSV 파일의 'path' 열에서 특정 접두사를 제거하고, 일치하는 파일을 새 폴더에 파일 이름만 복사합니다.
    """
    if not os.path.exists(index_path):
        print(f"❌ 오류: 인덱스 파일 경로를 찾을 수 없습니다: {index_path}")
        return

    if not os.path.isdir(target_dir):
        print(f"❌ 오류: 원본 대상 폴더 경로를 찾을 수 없습니다: {target_dir}")
        return
        
    # 새 대상 폴더 생성
    os.makedirs(new_target_dir, exist_ok=True)
    print(f"📁 새 대상 폴더 준비 완료: {new_target_dir}")
    print("-" * 30)

    # 1. 유지할 파일의 '상대 경로' 목록 로드
    print(f"🔍 인덱스 파일 로드 중: {index_path}")
    try:
        df = pd.read_csv(index_path)

        keep_paths_set = set()
        for p in df['path'].tolist():
            if pd.notna(p):
                clean_path = str(p).replace('\\', '/').replace(path_to_remove.replace('\\', '/'), '', 1)
                keep_paths_set.add(os.path.normpath(clean_path))
        
    except KeyError:
        print("❌ 오류: CSV 파일에 'path'라는 열이 없습니다. 열 이름이 정확한지 확인해 주세요.")
        return
    except Exception as e:
        print(f"❌ 인덱스 파일 읽기 중 오류 발생: {e}")
        return

    print(f"✅ 유지할 파일 경로 목록 로드 완료. 총 {len(keep_paths_set)}개의 경로.")
    print(f"💡 인덱스 (접두사 제거 후) 경로의 예시: {list(keep_paths_set)[:3]}") 
    print("-" * 30)


    # 2. 원본 폴더 순회 및 일치하는 파일 복사
    copied_count = 0
    
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            full_source_path = os.path.join(root, file)

            relative_path = os.path.normpath(
                os.path.relpath(full_source_path, target_dir)
                .replace('\\', '/')
            )

            if relative_path in keep_paths_set:
                try:
                    file_name = os.path.basename(full_source_path) 
                    full_destination_path = os.path.join(new_target_dir, file_name)

                    shutil.copy2(full_source_path, full_destination_path)
                    
                    print(f"📁 복사됨 (평탄화): {file_name}")
                    copied_count += 1
                except OSError as e:
                    print(f"❌ 파일 복사 실패 ({full_source_path}): {e}")

    print("-" * 30)
    print(f"🎉 **작업 완료**: 총 **{copied_count}개**의 파일이 새 폴더 ({new_target_dir})로 복사되었습니다. (하위 폴더 구조 제거됨)")


if __name__ == "__main__":
    INDEX_PATH = INDEX_FILE_PATH.replace('\\', '/')
    TARGET_PATH = TARGET_DIR.replace('\\', '/')
    NEW_TARGET_PATH = NEW_TARGET_DIR.replace('\\', '/')

    PATH_TO_REMOVE = BASE_PATH_TO_REMOVE.replace('\\', '/')

    copy_matching_files_flat_cleaned(INDEX_PATH, TARGET_PATH, NEW_TARGET_PATH, PATH_TO_REMOVE)