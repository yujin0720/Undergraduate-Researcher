import os
import numpy as np
import anndata as ad

# 경로 설정
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "../data/processed"))
output_dir = data_dir
os.makedirs(output_dir, exist_ok=True)

# anndata 불러오기
adata = ad.read_h5ad(os.path.join(data_dir, "adata_skmel28_top2000.h5ad"))

# 샘플 구분
is_pre = adata.obs["sample"].isin(["sample1", "sample2", "sample3", "sample4"])
is_post = adata.obs["sample"].isin(["sample5", "sample6", "sample7", "sample8"])

adata_pre = adata[is_pre].copy()
adata_post = adata[is_post].copy()

# 바코드 앞부분만 추출하여 매칭용으로 사용
pre_barcodes_full = adata_pre.obs_names
post_barcodes_full = adata_post.obs_names

pre_barcodes_clean = [x.split("_")[0].split("-")[0] for x in pre_barcodes_full]
post_barcodes_clean = [x.split("_")[0].split("-")[0] for x in post_barcodes_full]

# 교집합
matched_barcodes = np.intersect1d(pre_barcodes_clean, post_barcodes_clean)
print(f"📌 공통된 셀 수: {len(matched_barcodes)}")

# 매칭된 인덱스 가져오기
pre_idx = [i for i, bc in enumerate(pre_barcodes_clean) if bc in matched_barcodes]
post_idx = [i for i, bc in enumerate(post_barcodes_clean) if bc in matched_barcodes]

# X_pre, X_post 추출 및 저장
X_pre_matched = adata_pre.X[pre_idx].toarray().astype(np.float32)
X_post_matched = adata_post.X[post_idx].toarray().astype(np.float32)

np.save(os.path.join(output_dir, "X_pre_matched.npy"), X_pre_matched)
np.save(os.path.join(output_dir, "X_post_matched.npy"), X_post_matched)

print("✅ 공통 셀 기준 X_pre/X_post 저장 완료")
