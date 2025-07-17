import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# [1] 데이터 로드
X_pre = np.load("../data/processed/X_pre.npy")
X_post = np.load("../data/processed/X_post.npy")

# [2] 길이 맞추기 (1:1 대응 필요)
n = min(len(X_pre), len(X_post))
X_pre = X_pre[:n]
X_post = X_post[:n]

# [3] train/test 분할 (80:20)
X_pre_train, X_pre_test, X_post_train, X_post_test = train_test_split(
    X_pre, X_post, test_size=0.2, random_state=42
)

# [4] Tensor 변환
X_pre_train = torch.tensor(X_pre_train, dtype=torch.float32)
X_post_train = torch.tensor(X_post_train, dtype=torch.float32)
X_pre_test = torch.tensor(X_pre_test, dtype=torch.float32)
X_post_test = torch.tensor(X_post_test, dtype=torch.float32)

# [5] DataLoader 구성
train_loader = DataLoader(TensorDataset(X_pre_train, X_post_train), batch_size=256, shuffle=True)
test_loader = DataLoader(TensorDataset(X_pre_test, X_post_test), batch_size=256)

# [6] 개선된 MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
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

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# [7] 모델 초기화
input_dim = X_pre.shape[1]
output_dim = X_post.shape[1]
model = MLP(input_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# [8] 학습 루프
print("[Step 5] MLP 모델 학습 시작")
for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # [9] 평가 루프
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)

    print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

# [10] 모델 저장
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/mlp_pre_post.pt")
print("✅ 모델 저장 완료: models/mlp_pre_post.pt")
