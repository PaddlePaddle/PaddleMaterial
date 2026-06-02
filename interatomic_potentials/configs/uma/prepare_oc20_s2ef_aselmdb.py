from __future__ import annotations

import argparse
import json
import lzma
import os
import re
import zlib
from pathlib import Path

import lmdb
import numpy as np
from ase.data import atomic_numbers


_ENERGY_RE = re.compile(r"(?:^|\s)energy=([^\s]+)")
_LATTICE_RE = re.compile(r'Lattice="([^"]+)"')


def _iter_extxyz_frames(path: Path):
    with lzma.open(path, "rt") as fp:
        while True:
            line = fp.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            natoms = int(line)
            header = fp.readline().strip()
            energy_match = _ENERGY_RE.search(header)
            lattice_match = _LATTICE_RE.search(header)
            if energy_match is None or lattice_match is None:
                raise ValueError(f"Missing energy/lattice in {path}")
            energy = float(energy_match.group(1))
            cell = np.asarray(
                [float(x) for x in lattice_match.group(1).split()], dtype=np.float64
            ).reshape(3, 3)

            numbers = []
            positions = []
            tags = []
            forces = []
            for _ in range(natoms):
                parts = fp.readline().split()
                numbers.append(atomic_numbers[parts[0]])
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                tags.append(int(parts[5]))
                forces.append([float(parts[6]), float(parts[7]), float(parts[8])])

            yield {
                "numbers": numbers,
                "positions": positions,
                "cell": cell.tolist(),
                "pbc": [True, True, True],
                "tags": tags,
                "energy": energy,
                "forces": forces,
            }


def _write_aselmdb(path: Path, rows: list[dict]) -> None:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    map_size = max(1 << 30, len(rows) * 200_000)
    env = lmdb.open(
        str(path),
        subdir=False,
        map_size=map_size,
        lock=False,
        meminit=False,
    )
    try:
        with env.begin(write=True) as txn:
            txn.put(b"length", str(len(rows)).encode("ascii"))
            for idx, row in enumerate(rows, start=1):
                payload = json.dumps(row, separators=(",", ":")).encode("utf-8")
                txn.put(str(idx).encode("ascii"), zlib.compress(payload))
    finally:
        env.sync()
        env.close()


def _write_aselmdb_stream(path: Path, frames, total: int) -> int:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    map_size = max(1 << 30, total * 200_000)
    env = lmdb.open(
        str(path),
        subdir=False,
        map_size=map_size,
        lock=False,
        meminit=False,
    )
    count = 0
    try:
        with env.begin(write=True) as txn:
            for count, row in enumerate(frames, start=1):
                payload = json.dumps(row, separators=(",", ":")).encode("utf-8")
                txn.put(str(count).encode("ascii"), zlib.compress(payload))
            txn.put(b"length", str(count).encode("ascii"))
    finally:
        env.sync()
        env.close()
    if count != total:
        raise ValueError(f"Wrote {count} frames to {path}, expected {total}.")
    return count


def _sorted_extxyz_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("*.extxyz.xz"), key=lambda p: int(p.stem.split(".")[0]))
    if not files:
        raise ValueError(f"No *.extxyz.xz files found in {raw_dir}.")
    return files


def _iter_frames(raw_dir: Path, total: int, offset: int = 0):
    seen = 0
    yielded = 0
    files = _sorted_extxyz_files(raw_dir)
    for file_path in files:
        for frame in _iter_extxyz_frames(file_path):
            if seen < offset:
                seen += 1
                continue
            frame["data"] = {"sid": f"{file_path.name}:{seen}"}
            yield frame
            seen += 1
            yielded += 1
            if yielded >= total:
                return
    raise ValueError(
        f"Only found {yielded} frames in {raw_dir} after offset {offset}, "
        f"requested {total}."
    )


def _take_frames(raw_dir: Path, total: int) -> list[dict]:
    return list(_iter_frames(raw_dir, total))


def _write_metadata(split_dir: Path, source: str, num_samples: int) -> None:
    with open(split_dir / "metadata.json", "w") as fp:
        json.dump(
            {
                "source": source,
                "num_samples": num_samples,
                "format": "UMA LMDB JSON fallback",
            },
            fp,
            indent=2,
        )


def _write_split(
    out_dir: Path,
    split: str,
    raw_dir: Path,
    count: int,
    offset: int,
    source: str,
) -> None:
    split_dir = out_dir / split
    written = _write_aselmdb_stream(
        split_dir / f"{split}.aselmdb",
        _iter_frames(raw_dir, count, offset=offset),
        count,
    )
    _write_metadata(split_dir, source, written)
    print(f"Wrote {written} samples to {split_dir}/{split}.aselmdb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        default="./data/oc20/raw/s2ef_train_200K/s2ef_train_200K",
        help="Directory containing OC20 *.extxyz.xz files.",
    )
    parser.add_argument(
        "--val-raw-dir",
        default=None,
        help=(
            "Optional directory containing an official validation split. When set, "
            "val/test are sampled from this directory instead of from --raw-dir."
        ),
    )
    parser.add_argument(
        "--test-raw-dir",
        default=None,
        help=(
            "Optional directory for test samples. Defaults to --val-raw-dir when "
            "--val-raw-dir is set."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="./data/oc20/uma_aselmdb",
        help="Output directory for UMA-compatible *.aselmdb files.",
    )
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--test-offset", type=int, default=None)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if args.val_raw_dir is None:
        total = args.train + args.val + args.test
        rows = _take_frames(raw_dir, total)

        splits = {
            "train": rows[: args.train],
            "val": rows[args.train : args.train + args.val],
            "test": rows[args.train + args.val :],
        }
        for split, split_rows in splits.items():
            split_dir = out_dir / split
            _write_aselmdb(split_dir / f"{split}.aselmdb", split_rows)
            _write_metadata(split_dir, "OC20 S2EF train 200k", len(split_rows))
            print(f"Wrote {len(split_rows)} samples to {split_dir}/{split}.aselmdb")
        return

    val_raw_dir = Path(args.val_raw_dir)
    test_raw_dir = Path(args.test_raw_dir) if args.test_raw_dir is not None else val_raw_dir
    test_offset = args.test_offset if args.test_offset is not None else args.val_offset + args.val

    _write_split(
        out_dir,
        "train",
        raw_dir,
        args.train,
        args.train_offset,
        "OC20 S2EF train 200k",
    )
    _write_split(
        out_dir,
        "val",
        val_raw_dir,
        args.val,
        args.val_offset,
        "OC20 S2EF val_id",
    )
    _write_split(
        out_dir,
        "test",
        test_raw_dir,
        args.test,
        test_offset,
        "OC20 S2EF val_id",
    )


if __name__ == "__main__":
    main()
