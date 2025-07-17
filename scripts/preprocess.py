import scanpy as sc
import os

# 설정
DO_STEP_1_FILTER = False  # True로 바꾸면 처음부터 필터링 재실행
RAW_PATH = "../data/raw/pre-post_dataset.h5ad"
FILTERED_PATH = "../data/processed/adata_filtered.h5ad"
PROCESSED_PATH = "../data/processed/adata_processed_2000genes.h5ad"

os.makedirs("../data/processed", exist_ok=True)

# Step 1 & 2: 필터링 (optional)
if DO_STEP_1_FILTER:
    print("[Step 1] 원본 .h5ad 파일에서 normal/tumor만 필터링 중...")
    adata = sc.read_h5ad(RAW_PATH)

    # normal / tumor만 추출
    adata = adata[adata.obs["harm_sample.type"].isin(["normal", "tumor"])].copy()

    # 확인
    print(adata.obs["harm_sample.type"].value_counts())
    print("필터링 후 shape:", adata.shape)

    # 저장
    adata.write(FILTERED_PATH)
    print(f"✅ 필터링 완료 → 저장됨: {FILTERED_PATH}")
else:
    # 이미 필터링된 파일 불러오기
    print("[Step 1] 필터링된 파일 로딩 중...")
    adata = sc.read_h5ad(FILTERED_PATH)
    print("로드된 shape:", adata.shape)

# Step 3: 정규화 + log 변환 + 고변이 유전자 선택

# 1. 정규화
print("\n[Step 3] - (1) 정규화 중...")
sc.pp.normalize_total(adata, target_sum=1e4)

# 2. log 변환

print("\n[Step 3] - (2) 로그 변환 중...")
sc.pp.log1p(adata)


# 3. 고변이 유전자 선택 (2000개)
print("\n[Step 3] - (3) 고변이 유전자 선택 중...")
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

# 결과 저장
adata.write(PROCESSED_PATH)
print(f"✅ 정규화 및 유전자 선택 완료 → 저장됨: {PROCESSED_PATH}")
print("최종 shape:", adata.shape)
