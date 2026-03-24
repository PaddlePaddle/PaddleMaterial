#!/usr/bin/env python3
"""MatterGen PyTorch 采样子脚本 —— 在 matinvent conda 环境下执行。"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

RAW_ROOT = Path(sys.argv[1])
OUT_JSON = Path(sys.argv[2])
SEED = int(sys.argv[3])
NUM_SAMPLES = int(sys.argv[4])
NUM_INFERENCE_STEPS = int(sys.argv[5])

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(RAW_ROOT))

try:
    from huggingface_hub import hf_hub_download  # noqa: E402
    from mattergen.common.utils.data_classes import MatterGenCheckpointInfo  # noqa
    from models.mattergen.pl_module import MatterGenModule  # noqa
    from models.mattergen.sample import MatterGenSampler  # noqa

    # 替代 from_hf_hub：手动下载将由 matinvent_test.py 预先完成，此处为缓存命中
    ckpt_cache = hf_hub_download(
        repo_id="microsoft/mattergen",
        filename="checkpoints/mattergen_base/checkpoints/last.ckpt",
    )
    config_cache = hf_hub_download(
        repo_id="microsoft/mattergen",
        filename="checkpoints/mattergen_base/config.yaml",
    )
    ckpt_info = MatterGenCheckpointInfo(
        model_path=str(Path(config_cache).parent),
        load_epoch="last",
    )
    model = MatterGenModule.load_from_checkpoint_and_config(
        ckpt_info.checkpoint_path,
        config=ckpt_info.config.lightning_module,
        map_location=device,
        strict=False,
    )[0]
    model.eval()

    results = []
    for idx in range(NUM_SAMPLES):
        with torch.no_grad():
            try:
                sampler = MatterGenSampler(
                    batch_size=1,
                    num_batches=1,
                    sampling_config_overrides=[
                        "sampler_partial.n_steps_corrector=0",
                    ],
                )
                samples, _ = sampler.generate(
                    model=model,
                    batch_size=1,
                    num_batches=1,
                )
                for s in samples:
                    results.append(
                        {
                            "frac_coords": s.pos.cpu().numpy().tolist(),
                            "lattice": s.cell.cpu().numpy().tolist(),
                            "atom_types": s.atomic_numbers.cpu().numpy().tolist(),
                            "num_atoms": int(s.num_atoms.item()),
                        }
                    )
            except Exception as e:
                print(f"[PT] Sampling failed: {e}", flush=True)
                import traceback

                traceback.print_exc()

    with open(OUT_JSON, "w") as fp:
        json.dump({"samples": results, "num_samples": len(results)}, fp)
    print(f"[PT] Sampling done: {len(results)} samples", flush=True)

except Exception as e:
    print(f"[PT] MatterGen sampling failed: {e}", flush=True)
    import traceback

    traceback.print_exc()
