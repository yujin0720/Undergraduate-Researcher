import os
import numpy as np
import torch
from models.mlp import MLP_Deep

# 1. 경로 설정
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "../data/processed/A375P_h5ad"))
model_path = os.path.abspath(os.path.join(base_dir, "../models/mlp_a375p_v2.pt"))
output_path = os.path.join(data_dir, "X_post_pred_v2.npy")

# 2. 정규화된 데이터 불러오기
X_pre = np.load(os.path.join(data_dir, "X_pre_scaled.npy"))

# 3. 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_Deep(input_dim=X_pre.shape[1]).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 4. 예측
with torch.no_grad():
    X_pre_tensor = torch.tensor(X_pre, dtype=torch.float32).to(device)
    X_pred_tensor = model(X_pre_tensor)
    X_pred = X_pred_tensor.cpu().numpy()

# 5. 저장
np.save(output_path, X_pred)
print(f"✅ 예측 완료! 저장 경로: {output_path}")
