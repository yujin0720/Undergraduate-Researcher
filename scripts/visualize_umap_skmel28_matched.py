import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# 경로 설정
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "../data/processed"))

X_real = np.load(os.path.join(data_dir, "X_post_matched.npy"))
X_pred = np.load(os.path.join(data_dir, "X_post_pred_skmel28_matched.npy"))

# 로그 변환 + 정규화
# 로그 변환 + 정규화 전에 NaN 및 음수값 제거
X_pred = np.nan_to_num(X_pred, nan=0.0)  # NaN → 0
X_pred[X_pred < 0] = 0  # 음수값 → 0

# 로그 변환
X_real_log = np.log1p(X_real)
X_pred_log = np.log1p(X_pred)

# UMAP 차원 축소
reducer = umap.UMAP(random_state=42)
X_combined = np.concatenate([X_real_log, X_pred_log], axis=0)
X_umap = reducer.fit_transform(X_combined)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:len(X_real), 0], X_umap[:len(X_real), 1], s=10, c='blue', label='Real Post')
plt.scatter(X_umap[len(X_real):, 0], X_umap[len(X_real):, 1], s=10, c='orange', label='Predicted Post')
plt.legend()
plt.title("UMAP: Real vs Predicted Post-treatment Cells")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "umap_skmel28_matched.png"))
plt.show()

# 평가 지표
mse = mean_squared_error(X_real_log, X_pred_log)
cos_sim = cosine_similarity(X_real_log, X_pred_log).diagonal().mean()

print(f"📏 MSE: {mse:.4f}")
print(f"📏 평균 Cosine Similarity: {cos_sim:.4f}")
