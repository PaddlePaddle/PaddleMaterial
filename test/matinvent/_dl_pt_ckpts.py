#!/usr/bin/env python3
"""预下载 PyTorch 原版权重到指定目录，供 pt_runner 子进程直接加载。

在 matinvent conda 环境下执行（需要 huggingface_hub）。

用法:
    python _dl_pt_ckpts.py <target_dir> [--diffcsp] [--mattergen]
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description="预下载 PyTorch 原版 MatInvent 权重")
    parser.add_argument("target_dir", type=str, help="目标目录（权重存放位置）")
    parser.add_argument("--diffcsp", action="store_true", help="下载 DiffCSP 权重")
    parser.add_argument("--mattergen", action="store_true", help="下载 MatterGen 权重")
    args = parser.parse_args()

    dst = Path(args.target_dir)
    dst.mkdir(parents=True, exist_ok=True)

    if args.diffcsp:
        c = hf_hub_download(
            repo_id="jwchen25/MatInvent",
            filename="diffcsp_mp20/last.ckpt",
        )
        shutil.copy(c, dst / "raw-diffcsp.ckpt")
        print("[DL] raw-diffcsp.ckpt done", flush=True)

    if args.mattergen:
        c = hf_hub_download(
            repo_id="microsoft/mattergen",
            filename="checkpoints/mattergen_base/checkpoints/last.ckpt",
        )
        hf_hub_download(
            repo_id="microsoft/mattergen",
            filename="checkpoints/mattergen_base/config.yaml",
        )
        shutil.copy(c, dst / "raw-mattergen.ckpt")
        print("[DL] raw-mattergen.ckpt done", flush=True)


if __name__ == "__main__":
    main()
