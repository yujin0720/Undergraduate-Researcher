import os
import numpy as np
import umap
import matplotlib.pyplot as plt

# 현재 스크립트 위치 기준으로 상대경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # connects/

X_real_path = os.path.join(base_dir, "data", "processed", "X_post_wm989.npy")
X_pred_path = os.path.join(base_dir, "data", "processed", "X_post_pred_wm989.npy")
save_path = os.path.join(base_dir, "results", "umap_wm989.png")

# 데이터 로드
X_real = np.load(X_real_path)
X_pred = np.load(X_pred_path)

# UMAP 변환
reducer = umap.UMAP(random_state=42)
X_combined = np.concatenate([X_real, X_pred], axis=0)
embedding = reducer.fit_transform(X_combined)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:len(X_real), 0], embedding[:len(X_real), 1], label="True", alpha=0.5)
plt.scatter(embedding[len(X_real):, 0], embedding[len(X_real):, 1], label="Predicted", alpha=0.5)
plt.title("UMAP: True vs Predicted Post-treatment (WM989)")
plt.legend()
plt.savefig(save_path)
print(f"✅ Saved UMAP visualization to {save_path}")
