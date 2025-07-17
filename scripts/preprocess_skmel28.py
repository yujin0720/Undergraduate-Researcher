# scripts/preprocess_skmel28_top2000.py

import anndata as ad
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "../data/processed/adata_skmel28_top2000.h5ad")
adata = ad.read_h5ad(data_path)

# 샘플 분리
is_pre = adata.obs["sample"].isin(["sample1", "sample2", "sample3", "sample4"])
is_post = adata.obs["sample"].isin(["sample5", "sample6", "sample7", "sample8"])

X_pre = np.log1p(adata.X[is_pre].toarray())
X_post = np.log1p(adata.X[is_post].toarray())

# 정규화
scaler_pre = StandardScaler()
scaler_post = StandardScaler()
X_pre_scaled = scaler_pre.fit_transform(X_pre)
X_post_scaled = scaler_post.fit_transform(X_post)

np.save(os.path.join(base_dir, "../data/processed/X_pre_skmel28_2000.npy"), X_pre_scaled)
np.save(os.path.join(base_dir, "../data/processed/X_post_skmel28_2000.npy"), X_post_scaled)
