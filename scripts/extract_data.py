import os
import tarfile
import gzip
import shutil

def extract_tar(tar_path, dest_dir):
    """tar 파일을 압축 해제"""
    if not os.path.exists(tar_path):
        print(f"❌ {tar_path} 없음")
        return
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=dest_dir)
        print(f"✅ TAR 압축 해제 완료: {tar_path} → {dest_dir}")

def extract_gz(gz_path, dest_dir):
    """gz 파일을 압축 해제하여 같은 이름의 .csv/.tsv로 저장"""
    if not os.path.exists(gz_path):
        print(f"❌ {gz_path} 없음")
        return
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(gz_path)
    out_path = os.path.join(dest_dir, filename.replace(".gz", ""))
    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"✅ GZ 압축 해제 완료: {gz_path} → {out_path}")

# === 경로 설정 ===
BASE = "C:/Users/mnmny/Desktop/connects/data/raw"

# === WM989 TAR ===
extract_tar(
    tar_path=os.path.join(BASE, "WM989", "GSE279162_RAW.tar"),
    dest_dir=os.path.join(BASE, "WM989")
)

# === SKMEL28 TAR ===
extract_tar(
    tar_path=os.path.join(BASE, "SKMEL28", "GSE162045_RAW.tar"),
    dest_dir=os.path.join(BASE, "SKMEL28")
)

# === A375P CSV/TSV.GZ ===
extract_gz(
    gz_path=os.path.join(BASE, "A375P", "GSE247684_a375_all_C_to_T4.csv.gz"),
    dest_dir=os.path.join(BASE, "A375P")
)

extract_gz(
    gz_path=os.path.join(BASE, "A375P", "GSE247684_a375_metadata_g1.tsv.gz"),
    dest_dir=os.path.join(BASE, "A375P")
)
