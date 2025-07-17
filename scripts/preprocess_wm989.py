import scanpy as sc
import os

# 경로 설정
input_path = "/data/processed/wm989_raw_merged.h5ad"
output_path = "/data/processed/wm989_filtered.h5ad"

# 데이터 로딩
print(f"📂 Loading data from: {input_path}")
adata = sc.read_h5ad(input_path)

# 세포 및 유전자 필터링
print("🔍 Filtering cells and genes...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 미토콘드리아 유전자 비율 계산
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# 정규화 및 로그변환
print("🧪 Normalizing and log-transforming...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 고변이 유전자 추출 (2000개)
print("📈 Selecting highly variable genes...")
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

# 저장
print(f"💾 Saving processed data to: {output_path}")
adata.write(output_path)
print("✅ Preprocessing complete.")
