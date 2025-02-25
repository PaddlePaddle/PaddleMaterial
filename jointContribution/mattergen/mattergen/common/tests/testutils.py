from pathlib import Path

import paddle

from paddle_geometric.data import Batch


def get_mp_20_debug_batch() -> Batch:
    return paddle.load(path=str(Path(__file__).resolve().parent / "mp_20_debug_batch.pt"))
