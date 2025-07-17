import torch
import torch.nn.functional as F
import numpy as np
from models.vae import VAE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import umap
import matplotlib.pyplot as plt
import os

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로딩
X_pre = np.load("../data/processed/X_pre.npy")
X_post = np.load("../data/processed/X_post.npy")

# 샘플 수 맞추기
min_samples = min(X_pre.shape[0], X_post.shape[0])
X_pre = X_pre[:min_samples]
X_post = X_post[:min_samples]

# 텐서 변환
X_pre_tensor = torch.tensor(X_pre, dtype=torch.float32).to(device)

# 모델 로드
model = VAE(input_dim=X_pre.shape[1]).to(device)
model.load_state_dict(torch.load("../models/vae.pt", map_location=device))
model.eval()

# 예측
with torch.no_grad():
    X_post_pred_tensor, _, _ = model(X_pre_tensor)
    X_post_pred = X_post_pred_tensor.cpu().numpy()

# 평가 지표: 1:1 cosine similarity 평균으로 계산
cos_sim = np.mean([
    cosine_similarity(X_post_pred[i:i+1], X_post[i:i+1])[0][0]
    for i in range(len(X_post))
])
mse = mean_squared_error(X_post, X_post_pred)
print(f"Cosine Similarity (mean): {cos_sim:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# UMAP 시각화
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
embedding = reducer.fit_transform(np.concatenate([X_post, X_post_pred]))

n = X_post.shape[0]
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:n, 0], embedding[:n, 1], c='blue', label='Real Post', alpha=0.5, s=5)
plt.scatter(embedding[n:, 0], embedding[n:, 1], c='orange', label='Predicted Post', alpha=0.5, s=5)
plt.legend()
plt.title("UMAP: Real vs Predicted Post-treatment")
os.makedirs("../results", exist_ok=True)
plt.savefig("../results/vae_umap.png", dpi=300)
plt.show()
