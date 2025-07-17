import os
import pandas as pd
from scipy.io import mmread

data_dir = "C:/Users/mnmny/Desktop/connects/data/raw/WM989"
samples = ["GSM8562999_Naive1", "GSM8563002_Dabrafenib"]

for sample in samples:
    print(f"\n🔎 Sample: {sample}")
    sample_path = os.path.join(data_dir, sample)

    # 1. matrix.mtx
    matrix_path = os.path.join(sample_path, "matrix.mtx")
    matrix = mmread(matrix_path).tocsc()
    print(f"  🧬 matrix.mtx shape: {matrix.shape} (rows: genes, cols: cells)")

    # 2. features.tsv
    features_path = os.path.join(sample_path, "features.tsv")
    features = pd.read_csv(features_path, sep="\t", header=None)
    print(f"  🔬 features.tsv shape: {features.shape}")
    print(f"  🔬 features.tsv preview:\n{features.head()}")

    # 3. barcodes.tsv
    barcodes_path = os.path.join(sample_path, "barcodes.tsv")
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)
    print(f"  🧪 barcodes.tsv shape: {barcodes.shape}")
    print(f"  🧪 barcodes.tsv preview:\n{barcodes.head()}")
