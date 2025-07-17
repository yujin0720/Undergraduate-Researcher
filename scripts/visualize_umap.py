# scripts/visualize_umap.py
import numpy as np
import umap
import matplotlib.pyplot as plt
import os

# 데이터 로드
X_post = np.load("../data/processed/X_post.npy")
X_post_pred = np.load("../results/X_post_pred.npy")

# 동일한 길이로 자르기
n = min(len(X_post), len(X_post_pred))
X_post = X_post[:n]
X_post_pred = X_post_pred[:n]

# UMAP 통합
reducer = umap.UMAP(random_state=42)
combined = np.concatenate([X_post, X_post_pred], axis=0)
embedding = reducer.fit_transform(combined)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:n, 0], embedding[:n, 1], s=3, label="True Post", alpha=0.5)
plt.scatter(embedding[n:, 0], embedding[n:, 1], s=3, label="Predicted Post", alpha=0.5)
plt.title("UMAP of True vs Predicted Post-treatment States")
plt.legend()
os.makedirs("../results", exist_ok=True)
plt.savefig("../results/umap_post_vs_pred.png", dpi=300)
plt.show()
