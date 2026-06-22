import os
from pathlib import Path
from argparse import Namespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLAGS_use_cudnn"] = "0"
os.environ["FLAGS_use_cuda"] = "0"

import paddle

paddle.set_device("cpu")

from ppmat.models.gmtnet import GMTNet


class Data:
    pass


def main():
    cfg_dir = Path(__file__).resolve().parent
    weight_path = cfg_dir / "paddle_model.pdparams"

    args = Namespace(
        atom_input_features=92,
        edge_features=512,
        embedding_features=128,
        output_features=9,
        num_layers=2,
        target="dielectric",
        use_mask=False,
        reduce_cell=False,
    )

    model = GMTNet(args)

    state = paddle.load(str(weight_path))
    model.set_state_dict(state)
    model.eval()

    data = Data()
    data.x = paddle.randn([5, 92], dtype="float32")
    data.edge_index = paddle.to_tensor(
        [
            [0, 1, 2, 3, 4, 0, 1, 2],
            [1, 2, 3, 4, 0, 2, 3, 4],
        ],
        dtype="int64",
    )
    data.edge_attr = paddle.randn([8, 3], dtype="float32")
    data.batch = paddle.zeros([5], dtype="int64")

    # Do not use paddle.no_grad().
    # GradientBlock internally needs paddle.grad.
    out = model(data, feat_mask=None, equality=None)

    print("GMTNet smoke test passed.")
    print("Output shape:", list(out.shape))
    print("Output mean:", float(out.mean()))
    print("Output std:", float(out.std()))


if __name__ == "__main__":
    main()
