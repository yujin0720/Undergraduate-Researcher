import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.mlp import MLP_Deep_2

# 경로
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "../data/processed"))
model_path = os.path.join(base_dir, "../models/mlp_skmel28_matched.pt")

# 데이터 로드
X_pre = np.load(os.path.join(data_dir, "X_pre_matched.npy"))
X_post = np.load(os.path.join(data_dir, "X_post_matched.npy"))

# 스케일링 (log1p + StandardScaler)
scaler_pre = StandardScaler()
scaler_post = StandardScaler()
X_pre_scaled = scaler_pre.fit_transform(np.log1p(X_pre))
X_post_scaled = scaler_post.fit_transform(np.log1p(X_post))

# 텐서 변환
X_pre_tensor = torch.tensor(X_pre_scaled, dtype=torch.float32)
X_post_tensor = torch.tensor(X_post_scaled, dtype=torch.float32)

# train/test 분할
X_train, X_val, y_train, y_val = train_test_split(X_pre_tensor, X_post_tensor, test_size=0.2, random_state=42)

# 데이터셋 로더
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_Deep_2(input_dim=X_pre.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 학습
for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}/100 | Loss: {total_loss:.4f}")

# 저장
torch.save(model.state_dict(), model_path)
print(f"✅ 모델 저장 완료: {model_path}")
