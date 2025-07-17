import scanpy as sc
import os

# 샘플 경로 설정
naive_path = r"C:/Users/mnmny/Desktop/connects/data/raw/WM989/GSM8562999_Naive1"
treated_path = r"C:/Users/mnmny/Desktop/connects/data/raw/WM989/GSM8563002_Dabrafenib"

# Naive 샘플
adata_naive = sc.read_mtx(os.path.join(naive_path, "matrix.mtx")).T
adata_naive.var_names = [line.strip().split('\t')[0] for line in open(os.path.join(naive_path, "features.tsv"))]
adata_naive.obs_names = [line.strip() for line in open(os.path.join(naive_path, "barcodes.tsv"))]
adata_naive.obs["condition"] = "Naive"

# Treated 샘플
adata_treated = sc.read_mtx(os.path.join(treated_path, "matrix.mtx")).T
adata_treated.var_names = [line.strip().split('\t')[0] for line in open(os.path.join(treated_path, "features.tsv"))]
adata_treated.obs_names = [line.strip() for line in open(os.path.join(treated_path, "barcodes.tsv"))]
adata_treated.obs["condition"] = "Dabrafenib"

# 병합
adata_merged = adata_naive.concatenate(adata_treated, join='outer')

# 저장
output_path = r"C:/Users/mnmny/Desktop/connects/data/processed/wm989_raw_merged.h5ad"
adata_merged.write(output_path)

print(f"✅ 병합 완료! 저장 위치: {output_path}")
