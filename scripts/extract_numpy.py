#scripts/extract_numpy.py
import scanpy as sc
import numpy as np
import os

# 파일 경로
adata_path = "../data/processed/adata_processed_2000genes.h5ad"
output_dir = "../data/processed/"
os.makedirs(output_dir, exist_ok=True)

# 데이터 로드
adata = sc.read_h5ad(adata_path)

# 상태 분리
condition = adata.obs["harm_sample.type"].values
X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

X_pre = X[condition == "normal"]
X_post = X[condition == "tumor"]

# 저장
np.save(f"{output_dir}/X_pre.npy", X_pre)
np.save(f"{output_dir}/X_post.npy", X_post)

print(f"✅ X_pre 저장: {X_pre.shape}")
print(f"✅ X_post 저장: {X_post.shape}")
