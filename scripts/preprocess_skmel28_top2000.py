import os
import numpy as np
import anndata as ad

# 경로 설정
base_dir = os.path.dirname(__file__)
input_path = os.path.join(base_dir, "../data/processed/adata_skmel28_top2000.h5ad")
output_dir = os.path.join(base_dir, "../data/processed")
os.makedirs(output_dir, exist_ok=True)

# anndata 불러오기
adata = ad.read_h5ad(input_path)

# 샘플 분리
is_pre = adata.obs["sample"].isin(["sample1", "sample2", "sample3", "sample4"])
is_post = adata.obs["sample"].isin(["sample5", "sample6", "sample7", "sample8"])

X_pre = adata.X[is_pre].toarray().astype(np.float32)
X_post = adata.X[is_post].toarray().astype(np.float32)

# 저장
np.save(os.path.join(output_dir, "X_pre_top2000.npy"), X_pre)
np.save(os.path.join(output_dir, "X_post_top2000.npy"), X_post)

print("✅ top2000 기반 X_pre, X_post 저장 완료")
