from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# 경로 설정
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "../data/processed"))

# 데이터 불러오기
X_real = np.load(os.path.join(data_dir, "X_post_top2000.npy"))
X_pred = np.load(os.path.join(data_dir, "X_post_pred_skmel28_top2000.npy"))
min_samples = min(X_real.shape[0], X_pred.shape[0])
X_real = X_real[:min_samples]
X_pred = X_pred[:min_samples]

# 지표 계산
mse = mean_squared_error(X_real, X_pred)
cosine = cosine_similarity(X_real, X_pred).diagonal().mean()

print(f"📏 MSE: {mse:.4f}")
print(f"📏 평균 Cosine Similarity: {cosine:.4f}")
