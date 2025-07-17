import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.mlp import MLP_Deep  # 개선된 MLP 구조

# 1. 경로 설정
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "../data/processed/A375P_h5ad"))
model_dir = os.path.abspath(os.path.join(base_dir, "../models"))
os.makedirs(model_dir, exist_ok=True)

# 2. 데이터 불러오기
X_pre = np.load(os.path.join(data_dir, "X_pre.npy"))
X_post = np.load(os.path.join(data_dir, "X_post.npy"))

min_samples = min(X_pre.shape[0], X_post.shape[0])
X_pre = X_pre[:min_samples]
X_post = X_post[:min_samples]


# 3. 정규화 (StandardScaler)
scaler_pre = StandardScaler()
scaler_post = StandardScaler()

X_pre_scaled = scaler_pre.fit_transform(X_pre)
X_post_scaled = scaler_post.fit_transform(X_post)

# 저장 (선택)
np.save(os.path.join(data_dir, "X_pre_scaled.npy"), X_pre_scaled)
np.save(os.path.join(data_dir, "X_post_scaled.npy"), X_post_scaled)

# 4. Tensor 변환
X_pre_tensor = torch.tensor(X_pre_scaled, dtype=torch.float32)
X_post_tensor = torch.tensor(X_post_scaled, dtype=torch.float32)

dataset = TensorDataset(X_pre_tensor, X_post_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 5. 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_Deep(input_dim=X_pre.shape[1]).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()
cos_sim = nn.CosineSimilarity(dim=1)

# 6. 학습
epochs = 30
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_Y in dataloader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        optimizer.zero_grad()
        pred = model(batch_X)

        loss_mse = mse_loss(pred, batch_Y)
        loss_cos = 1 - cos_sim(pred, batch_Y).mean()
        loss = 0.5 * loss_mse + 0.5 * loss_cos

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

# 7. 모델 저장
model_path = os.path.join(model_dir, "mlp_a375p_v2.pt")
torch.save(model.state_dict(), model_path)
print(f"✅ 모델 저장 완료: {model_path}")
