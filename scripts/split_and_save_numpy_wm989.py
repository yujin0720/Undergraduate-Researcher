import scanpy as sc
import numpy as np
import os

# 경로 설정
input_path = "C:/Users/mnmny/Desktop/connects/data/processed/wm989_filtered.h5ad"
output_dir = "C:/Users/mnmny/Desktop/connects/data/processed"
os.makedirs(output_dir, exist_ok=True)

# 파일 로딩
print(f"📂 Loading processed data: {input_path}")
adata = sc.read_h5ad(input_path)

# 샘플 분리
print("🔍 Splitting by treatment label...")
adata_naive = adata[adata.obs['batch'] == '0']  # Naive
adata_treat = adata[adata.obs['batch'] == '1']  # Dabrafenib

# 샘플 수 맞추기
min_cells = min(adata_naive.n_obs, adata_treat.n_obs)
X_pre = adata_naive.X[:min_cells].toarray() if hasattr(adata_naive.X, "toarray") else adata_naive.X[:min_cells]
X_post = adata_treat.X[:min_cells].toarray() if hasattr(adata_treat.X, "toarray") else adata_treat.X[:min_cells]

# 저장
np.save(os.path.join(output_dir, "X_pre_wm989.npy"), X_pre)
np.save(os.path.join(output_dir, "X_post_wm989.npy"), X_post)

print("✅ Saved:")
print(f" - X_pre_wm989.npy shape: {X_pre.shape}")
print(f" - X_post_wm989.npy shape: {X_post.shape}")
