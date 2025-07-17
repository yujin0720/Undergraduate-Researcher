import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.vae import VAE
import os

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 불러오기
X_pre = np.load("../data/processed/X_pre.npy")
X_post = np.load("../data/processed/X_post.npy")

# 샘플 수 맞추기
min_samples = min(X_pre.shape[0], X_post.shape[0])
X_pre = X_pre[:min_samples]
X_post = X_post[:min_samples]

# 텐서로 변환
X_pre_tensor = torch.tensor(X_pre, dtype=torch.float32)
X_post_tensor = torch.tensor(X_post, dtype=torch.float32)

# 데이터셋 및 로더 구성
dataset = TensorDataset(X_pre_tensor, X_post_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 모델 초기화
model = VAE(input_dim=X_pre.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 손실 함수 정의
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_div, recon_loss, kl_div

# 학습 루프
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_pre, batch_post in dataloader:
        batch_pre = batch_pre.to(device)
        batch_post = batch_post.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(batch_pre)
        loss, recon_loss, kl_loss = loss_function(recon, batch_post, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss:.4f}")

# 모델 저장
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "../models/vae.pt")
