# scripts/evaluate_model.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import os

# 데이터 로드
X_pre = np.load("../data/processed/X_pre.npy")
X_post = np.load("../data/processed/X_post.npy")

# 동일한 수로 자르기
n = min(len(X_pre), len(X_post))
X_pre = X_pre[:n]
X_post = X_post[:n]

# Tensor 변환
X_pre_tensor = torch.tensor(X_pre, dtype=torch.float32)

# 모델 정의
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.net(x)

# 모델 로드
input_dim = X_pre.shape[1]
model = MLP(input_dim)
model.load_state_dict(torch.load("../models/mlp_pre_post.pt"))
model.eval()

# 예측 수행
with torch.no_grad():
    X_post_pred_tensor = model(X_pre_tensor)
X_post_pred = X_post_pred_tensor.numpy()

# 예측 결과 저장 (UMAP용)
os.makedirs("../results", exist_ok=True)
np.save("../results/X_post_pred.npy", X_post_pred)

# Cosine 유사도
def compute_cosine_pairwise(x1, x2):
    x1_norm = x1 / np.linalg.norm(x1, axis=1, keepdims=True)
    x2_norm = x2 / np.linalg.norm(x2, axis=1, keepdims=True)
    return np.sum(x1_norm * x2_norm, axis=1)

pairwise_cos = compute_cosine_pairwise(X_post, X_post_pred)
mean_cosine = np.mean(pairwise_cos)

# 평가
mse = mean_squared_error(X_post, X_post_pred)
print(f"✅ 평균 Cosine Similarity (1:1): {mean_cosine:.4f}")
print(f"✅ 평균 MSE: {mse:.4f}")
