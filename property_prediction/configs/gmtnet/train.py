import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLAGS_use_cudnn"] = "0"
os.environ["FLAGS_use_cuda"] = "0"
import paddle
paddle.set_device('cpu')
import pickle
from pathlib import Path
import numpy as np
from paddle.io import Dataset, DataLoader

class TensorDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, _, _, _, _, _, labels = self.data[idx]
        labels = labels.reshape([-1])
        return paddle.to_tensor(x), paddle.to_tensor(labels)

data_path = Path(__file__).resolve().parent / "paddle_dielectric_data.pkl"
if data_path.exists():
    with open(data_path, "rb") as f:
        all_data = pickle.load(f)
else:
    rng = np.random.default_rng(42)
    all_data = []
    for _ in range(10):
        x = rng.normal(size=(4, 92)).astype("float32")
        labels = rng.normal(size=(3, 3)).astype("float32")
        all_data.append((x, None, None, None, None, None, labels))
    print(f"[WARN] {data_path} not found, using synthetic smoke-test data.")


total = len(all_data)
train_size = int(0.8 * total)
test_size = total - train_size

train_data = all_data[:train_size]
test_data = all_data[train_size:]

train_loader = DataLoader(TensorDataset(train_data), batch_size=1, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data), batch_size=1, shuffle=False)

class MinimalModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(92, 9)
        paddle.nn.initializer.KaimingUniform(negative_slope=0.01)(self.fc.weight)
        paddle.nn.initializer.Constant(0.0)(self.fc.bias)

    def forward(self, x):
        x_mean = x.mean(axis=1)
        return self.fc(x_mean)

model = MinimalModel()
optimizer = paddle.optimizer.Adam(learning_rate=1e-5, parameters=model.parameters())
criterion = paddle.nn.MSELoss()

for epoch in range(10):
    total_loss = 0.0
    for x, labels in train_loader:
        optimizer.clear_grad()
        out = model(x)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}")

model.eval()
test_mae = 0.0
with paddle.no_grad():
    for x, labels in test_loader:
        out = model(x)
        test_mae += paddle.nn.functional.l1_loss(out, labels, reduction='sum').item()
test_mae /= len(test_loader.dataset)
print(f"Test MAE: {test_mae:.6f}")
