import os
import anndata as ad
import scanpy as sc
import numpy as np

# 경로 설정
base_dir = os.path.dirname(__file__)
input_path = os.path.join(base_dir, "../data/processed/SKMEL28_h5ad/SKMEL28_full.h5ad")
output_path = os.path.join(base_dir, "../data/processed/adata_skmel28_top2000.h5ad")

# anndata 로드
adata = ad.read_h5ad(input_path)

# 로그 변환
adata.X = np.log1p(adata.X)

# 고변이 유전자 2000개 선택
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")

# 필터링 적용
adata = adata[:, adata.var["highly_variable"]]

# 저장
adata.write(output_path)
print(f"✅ 저장 완료: {output_path}")
