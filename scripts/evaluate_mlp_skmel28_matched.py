import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from models.mlp import MLP_Deep_2

# 경로
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "../data/processed"))
model_path = os.path.join(base_dir, "../models/mlp_skmel28_matched.pt")

# 데이터 로드
X_pre = np.load(os.path.join(data_dir, "X_pre_matched.npy"))
X_post = np.load(os.path.join(data_dir, "X_post_matched.npy"))

# 스케일링
scaler_pre = StandardScaler()
scaler_post = StandardScaler()
X_pre_scaled = scaler_pre.fit_transform(np.log1p(X_pre))
X_post_scaled = scaler_post.fit_transform(np.log1p(X_post))

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_Deep_2(input_dim=X_pre.shape[1]).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 예측
with torch.no_grad():
    X_pre_tensor = torch.tensor(X_pre_scaled, dtype=torch.float32).to(device)
    X_pred = model(X_pre_tensor).cpu().numpy()

# 저장
np.save(os.path.join(data_dir, "X_post_pred_skmel28_matched.npy"), X_pred)
print("✅ 예측 결과 저장 완료")
