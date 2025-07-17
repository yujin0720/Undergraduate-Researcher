import os
import numpy as np
import umap
import matplotlib.pyplot as plt

# 1. 경로 설정
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "../data/processed/A375P_h5ad"))

X_post_real = np.load(os.path.join(data_dir, "X_post_scaled.npy"))
X_post_pred = np.load(os.path.join(data_dir, "X_post_pred_v2.npy"))

# 2. 샘플 수 맞추기
min_samples = min(X_post_real.shape[0], X_post_pred.shape[0])
X_post_real = X_post_real[:min_samples]
X_post_pred = X_post_pred[:min_samples]

# 3. UMAP 임베딩
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
X_all = np.vstack([X_post_real, X_post_pred])
embedding = reducer.fit_transform(X_all)

# 4. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:min_samples, 0], embedding[:min_samples, 1], c="blue", label="Real", alpha=0.5, s=10)
plt.scatter(embedding[min_samples:, 0], embedding[min_samples:, 1], c="red", label="Predicted", alpha=0.5, s=10)
plt.legend()
plt.title("UMAP v2: Real vs Predicted Post-treatment (A375P, Scaled)")
plt.tight_layout()

# 5. 저장
output_path = os.path.join(data_dir, "umap_a375p_v2.png")
plt.savefig(output_path, dpi=300)
print(f"✅ UMAP 시각화 v2 저장 완료: {output_path}")
