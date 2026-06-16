import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLAGS_use_cudnn"] = "0"
os.environ["FLAGS_use_cuda"] = "0"
import paddle
paddle.set_device('cpu')
import pickle
import numpy as np
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "ppmat" / "models" / "gmtnet"))

from gmtnet import GMTNet

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
test_data = all_data[int(0.8*total):]

class Args:
    pass
args = Args()
args.target = 'dielectric'
args.reduce_cell = False
args.use_mask = False

model = GMTNet(args)
model.eval()

test_mae = 0.0
test_fnorm = 0.0
n_test = 0
for item in test_data:
    x, _, _, _, _, _, labels = item
    x_tensor = paddle.to_tensor(x)
    labels_tensor = paddle.to_tensor(labels.reshape([-1]))
    class Data:
        def __init__(self, x):
            self.x = x
    data = Data(x_tensor)
    feat_mask = paddle.zeros([1,1,1], dtype='float32')
    equality = paddle.zeros([x.shape[0], x.shape[0]], dtype='bool')
    with paddle.no_grad():
        out = model(data, feat_mask, equality)
    test_mae += paddle.nn.functional.l1_loss(out, labels_tensor, reduction='sum').item()
    pred_mat = out.reshape([3,3])
    label_mat = labels_tensor.reshape([3,3])
    fnorm = paddle.norm(pred_mat - label_mat, p='fro').item()
    test_fnorm += fnorm
    n_test += 1

test_mae /= n_test
test_fnorm /= n_test
print(f"Test MAE: {test_mae:.6f}")
print(f"Test Fnorm: {test_fnorm:.6f}")
