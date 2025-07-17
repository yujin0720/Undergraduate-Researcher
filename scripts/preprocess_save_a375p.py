import os
import pandas as pd
import numpy as np
import anndata as ad

# 1. 경로 설정
base_dir = os.path.dirname(__file__)
raw_data_dir = os.path.abspath(os.path.join(base_dir, "../data/raw/A375P"))
output_dir = os.path.abspath(os.path.join(base_dir, "../data/processed/A375P_h5ad"))
os.makedirs(output_dir, exist_ok=True)

# 2. 파일 경로
expr_path = os.path.join(raw_data_dir, "GSE247684_a375_all_C_to_T4.csv")
meta_path = os.path.join(raw_data_dir, "GSE247684_a375_metadata_g1.tsv")

# 3. 파일 로드
expr_df = pd.read_csv(expr_path, index_col=0)         # (genes × cells)
meta_df = pd.read_csv(meta_path, sep='\t')            # (cells × metadata)

# 4. 세포 이름 정리 (일치 여부 확인)
expr_cells = expr_df.columns.tolist()
meta_cells = meta_df['cell'].tolist()

# 5. 교집합 셀만 사용 (안전 처리)
common_cells = list(set(expr_cells) & set(meta_cells))
expr_df = expr_df[common_cells]
meta_df = meta_df[meta_df['cell'].isin(common_cells)]
meta_df = meta_df.set_index('cell').loc[expr_df.columns]  # 순서 정렬

# 6. h5ad 생성
adata = ad.AnnData(X=expr_df.T.values)  # shape = (cells × genes)
adata.obs = meta_df
adata.var_names = expr_df.index
adata.obs_names = expr_df.columns

# 7. 저장
adata_path = os.path.join(output_dir, "A375P_full.h5ad")
adata.write(adata_path)
print(f"✅ h5ad 저장 완료: {adata_path}")

# 8. pre/post 분리 기준: 'orig.ident' or cell 이름의 suffix (_C, _T4)
is_post = adata.obs_names.str.endswith("_T4") | (adata.obs['orig.ident'] == 'T4')
is_pre = ~is_post

X_pre = adata.X[is_pre]
X_post = adata.X[is_post]

# 9. npy 저장
np.save(os.path.join(output_dir, "X_pre.npy"), X_pre)
np.save(os.path.join(output_dir, "X_post.npy"), X_post)
print("✅ X_pre.npy / X_post.npy 저장 완료")
