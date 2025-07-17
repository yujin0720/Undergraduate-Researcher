# scripts/explore_data.py
import scanpy as sc

# 데이터 로드
adata = sc.read_h5ad("../data/raw/pre-post_dataset.h5ad")
print(adata)

# 메타데이터 컬럼 확인
print("obs.columns:")
print(adata.obs.columns)

# 상태 분포 보기
print("harm_sample.type:")
print(adata.obs["harm_sample.type"].value_counts())
