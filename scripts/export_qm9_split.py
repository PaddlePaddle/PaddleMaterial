#!/usr/bin/env python3
"""Export QM9 into train.csv, val.csv, and test.csv.

Output rows are one molecule each. The split policy matches the earlier export:
remove Figshare's 3,054 uncharacterized molecule IDs, then shuffle the remaining
130,831 molecule IDs with seed 42 and split as 110000/10000/10831.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import urllib.request
from pathlib import Path


UNCHARACTERIZED_URL = "https://ndownloader.figshare.com/files/3195404"

PROPERTY_COLUMNS = [
    "A",
    "B",
    "C",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
]

HEADER = [
    "file_name",
    "raw_file_content",
    "standard_xyz",
    "mulliken_xyz",
    "molecule_id",
    "num_atoms",
    *PROPERTY_COLUMNS,
    "vibrational_frequencies",
    "canonical_smiles",
    "isomeric_smiles",
    "canonical_inchi",
    "isomeric_inchi",
]


def natural_key(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot infer molecule id from {path.name}")
    return int(match.group(1))


def parse_float_text(value: str) -> str:
    """Normalize rare Mathematica-style exponents while keeping CSV text stable."""
    return str(float(value.replace("*^", "e")))


def parse_xyz_coord_text(value: str) -> str:
    """Format XYZ coordinates as plain decimals for RDKit compatibility."""
    return f"{float(value.replace('*^', 'e')):.10f}"


def parse_charge_text(value: str) -> str:
    """Format Mulliken partial charges as plain decimals."""
    return f"{float(value.replace('*^', 'e')):.10f}"


def ensure_uncharacterized(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(UNCHARACTERIZED_URL, path)
    return path


def load_bad_ids(path: Path) -> set[int]:
    bad = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            bad.add(int(parts[0]))
    if len(bad) != 3054:
        raise ValueError(f"Expected 3054 uncharacterized ids, got {len(bad)}")
    return bad


def split_ids(ids: list[int], seed: int) -> dict[int, str]:
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    train = set(shuffled[:110_000])
    val = set(shuffled[110_000:120_000])
    return {
        molecule_id: "train"
        if molecule_id in train
        else "val"
        if molecule_id in val
        else "test"
        for molecule_id in ids
    }


def parse_qm9_file(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    num_atoms = int(lines[0].strip())

    props = lines[1].split()
    if len(props) < 17:
        raise ValueError(f"Malformed property line in {path.name}")
    molecule_id = int(props[1])
    if molecule_id != natural_key(path):
        raise ValueError(f"{path.name} contains molecule id {molecule_id}")

    property_values = dict(zip(PROPERTY_COLUMNS, [parse_float_text(x) for x in props[2:17]]))
    comment_line = f"gdb {molecule_id}"

    standard_atom_lines = []
    mulliken_atom_lines = []
    for line in lines[2 : 2 + num_atoms]:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed atom line in {path.name}: {line!r}")
        element = parts[0]
        x, y, z = [parse_xyz_coord_text(x) for x in parts[1:4]]
        charge = parse_charge_text(parts[4])
        standard_atom_lines.append(f"{element}\t{x}\t{y}\t{z}")
        mulliken_atom_lines.append(f"{element}\t{x}\t{y}\t{z}\t{charge}")

    standard_xyz = "\n".join([str(num_atoms), comment_line, *standard_atom_lines])
    mulliken_xyz = "\n".join([str(num_atoms), comment_line, *mulliken_atom_lines])

    freq_idx = 2 + num_atoms
    frequencies = " ".join(parse_float_text(x) for x in lines[freq_idx].split())
    smiles = lines[freq_idx + 1].split()
    inchi = lines[freq_idx + 2].split()

    return {
        "file_name": path.name,
        "raw_file_content": raw,
        "standard_xyz": standard_xyz,
        "mulliken_xyz": mulliken_xyz,
        "molecule_id": molecule_id,
        "num_atoms": num_atoms,
        **property_values,
        "vibrational_frequencies": frequencies,
        "canonical_smiles": smiles[0] if len(smiles) > 0 else "",
        "isomeric_smiles": smiles[1] if len(smiles) > 1 else "",
        "canonical_inchi": inchi[0] if len(inchi) > 0 else "",
        "isomeric_inchi": inchi[1] if len(inchi) > 1 else "",
    }


def export(input_dir: Path, output_dir: Path, uncharacterized: Path, seed: int) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bad_ids = load_bad_ids(ensure_uncharacterized(uncharacterized))
    files = sorted(input_dir.glob("*.xyz"), key=natural_key)
    ids = [natural_key(path) for path in files if natural_key(path) not in bad_ids]
    split_lookup = split_ids(ids, seed)

    handles = {}
    writers = {}
    counts = {"train": 0, "val": 0, "test": 0}
    try:
        for split in ["train", "val", "test"]:
            handle = (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=HEADER)
            writer.writeheader()
            handles[split] = handle
            writers[split] = writer

        for path in files:
            molecule_id = natural_key(path)
            split = split_lookup.get(molecule_id)
            if split is None:
                continue
            writers[split].writerow(parse_qm9_file(path))
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("./qm9.xyz"))
    parser.add_argument("--output-dir", type=Path, default=Path("./qm9_split"))
    parser.add_argument(
        "--uncharacterized",
        type=Path,
        default=Path("./qm9_split_scripts/uncharacterized.txt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = export(args.input_dir, args.output_dir, args.uncharacterized, args.seed)
    print(counts)


if __name__ == "__main__":
    main()
