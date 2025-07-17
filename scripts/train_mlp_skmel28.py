import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.mlp import MLP_Deep_2
import anndata as ad

# 1. 경로 설정
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "../data/processed/SKMEL28_h5ad"))
model_dir = os.path.abspath(os.path.join(base_dir, "../models"))
os.makedirs(model_dir, exist_ok=True)

# 2. anndata 불러오기
adata_path = os.path.join(data_dir, "SKMEL28_full.h5ad")
adata_all = ad.read_h5ad(adata_path)

# 3. pre/post 분리
is_pre = adata_all.obs["sample"].isin(["sample1", "sample2", "sample3", "sample4"])
is_post = adata_all.obs["sample"].isin(["sample5", "sample6", "sample7", "sample8"])

X_pre = adata_all.X[is_pre].toarray().astype(np.float32)
X_post = adata_all.X[is_post].toarray().astype(np.float32)

# 4. 로그 변환 + 정규화
X_pre = np.log1p(X_pre)
X_post = np.log1p(X_post)

scaler_pre = StandardScaler()
scaler_post = StandardScaler()

X_pre_scaled = scaler_pre.fit_transform(X_pre)
X_post_scaled = scaler_post.fit_transform(X_post)

# 5. 샘플 수 맞추기
min_samples = min(X_pre_scaled.shape[0], X_post_scaled.shape[0])
X_pre_scaled = X_pre_scaled[:min_samples]
X_post_scaled = X_post_scaled[:min_samples]

print("🔍 X_pre mean/std:", X_pre_scaled.mean(), X_pre_scaled.std())
print("🔍 X_post mean/std:", X_post_scaled.mean(), X_post_scaled.std())


# (선택) 저장
np.save(os.path.join(data_dir, "X_pre_scaled.npy"), X_pre_scaled)
np.save(os.path.join(data_dir, "X_post_scaled.npy"), X_post_scaled)

# 6. Tensor 변환
X_pre_tensor = torch.tensor(X_pre_scaled, dtype=torch.float32)
X_post_tensor = torch.tensor(X_post_scaled, dtype=torch.float32)

dataset = TensorDataset(X_pre_tensor, X_post_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 7. 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_Deep_2(input_dim=X_pre_tensor.shape[1]).to(device)

mse_loss = nn.MSELoss()  # ✅ 이 줄 추가해야 함!
loss = nn.MSELoss()
# 8. 학습 루프
epochs = 30
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
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
model_path = os.path.join(model_dir, "mlp_skmel28.pt")
torch.save(model.state_dict(), model_path)
print(f"✅ 모델 저장 완료: {model_path}")
