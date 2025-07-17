import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.mlp import MLP_Deep_2

# 1. 경로 설정
base_dir = os.path.dirname(__file__)

model_dir = os.path.abspath(os.path.join(base_dir, "../models"))
os.makedirs(model_dir, exist_ok=True)

# 2. top2000 기반 데이터 불러오기
data_dir = os.path.abspath(os.path.join(base_dir, "../data/processed"))
X_pre = np.load(os.path.join(data_dir, "X_pre_top2000.npy"))
X_post = np.load(os.path.join(data_dir, "X_post_top2000.npy"))

# 3. 샘플 수 맞추기
min_samples = min(X_pre.shape[0], X_post.shape[0])
X_pre = X_pre[:min_samples]
X_post = X_post[:min_samples]

# 4. 통계 출력
print("🔍 X_pre mean/std:", X_pre.mean(), X_pre.std())
print("🔍 X_post mean/std:", X_post.mean(), X_post.std())

# 5. Tensor 변환
X_pre_tensor = torch.tensor(X_pre, dtype=torch.float32)
X_post_tensor = torch.tensor(X_post, dtype=torch.float32)

dataset = TensorDataset(X_pre_tensor, X_post_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 6. 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_Deep_2(input_dim=X_pre_tensor.shape[1]).to(device)

# 7. 손실함수 & 옵티마이저
mse_loss = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

# 8. 학습 루프
epochs = 100
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_Y in dataloader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        optimizer.zero_grad()
        pred = model(batch_X)
        loss = mse_loss(pred, batch_Y)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

# 9. 모델 저장
model_path = os.path.join(model_dir, "mlp_skmel28_top2000.pt")
torch.save(model.state_dict(), model_path)
print(f"✅ 모델 저장 완료: {model_path}")
