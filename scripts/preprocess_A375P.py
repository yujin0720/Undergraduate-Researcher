import pandas as pd
import os

# 상대 경로 설정
base_dir = os.path.dirname(__file__)
raw_data_dir = os.path.abspath(os.path.join(base_dir, "../data/raw/A375P"))

# ✔️ 실제 확장자 명시
csv_path = os.path.join(raw_data_dir, "GSE247684_a375_all_C_to_T4.csv")
metadata_path = os.path.join(raw_data_dir, "GSE247684_a375_metadata_g1.tsv")

print("📁 raw_data_dir:", raw_data_dir)
print("📄 csv_path:", csv_path)
print("📄 metadata_path:", metadata_path)

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ 파일 없음: {csv_path}")
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"❌ 파일 없음: {metadata_path}")

# 📥 데이터 불러오기
expr_df = pd.read_csv(csv_path)
meta_df = pd.read_csv(metadata_path, sep="\t")

# 🔍 상위 미리보기
print("📊 Expression Data:")
print(expr_df.head())

print("\n🧬 Metadata:")
print(meta_df.head())
