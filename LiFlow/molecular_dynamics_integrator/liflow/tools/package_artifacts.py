# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""Build the reproducible LiFlow model archive and its manifest.

Produces a POSIX-layout zip (entries use '/' separators) plus a manifest JSON
with byte sizes and SHA256 per entry, then verifies a fresh extract round-trips
byte-for-byte. The archive is meant to be uploaded to a stable BCE URL later;
this script only ever produces and verifies the local artifact.

Layout (per the PR acceptance contract):
    liflow_universal/
    ├── liflow_universal_inference.yaml
    └── checkpoints/
        ├── propagator.pdparams
        └── corrector.pdparams
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
TASK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(config_yaml, propagator, corrector, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    staging = tempfile.mkdtemp(prefix="liflow_pkg_")
    prefix = os.path.join(staging, "liflow_universal")
    ckpt_dir = os.path.join(prefix, "checkpoints")
    os.makedirs(ckpt_dir)

    shutil.copyfile(config_yaml, os.path.join(prefix, "liflow_universal_inference.yaml"))
    shutil.copyfile(propagator, os.path.join(ckpt_dir, "propagator.pdparams"))
    shutil.copyfile(corrector, os.path.join(ckpt_dir, "corrector.pdparams"))

    archive_path = os.path.join(out_dir, "liflow_universal.zip")
    entries = {}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for dirpath, _dirs, files in os.walk(prefix):
            for name in sorted(files):
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, staging).replace(os.sep, "/")
                with open(full, "rb") as f:
                    data = f.read()
                zf.writestr(zipfile.ZipInfo(rel), data)
                entries[rel] = {"size": len(data), "sha256": sha256_bytes(data)}
    shutil.rmtree(staging, ignore_errors=True)

    manifest = {
        "name": "liflow_universal",
        "layout": {
            "liflow_universal_inference.yaml": "model configuration for inference",
            "checkpoints/propagator.pdparams": "P_universal conversion",
            "checkpoints/corrector.pdparams": "C_universal conversion",
        },
        "archive": os.path.abspath(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "entries": entries,
        "posix_separators": all("/" in k and "\\" not in k for k in entries),
    }
    manifest_path = os.path.join(out_dir, "liflow_universal_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # --- verify: fresh extract round-trips byte-for-byte ---
    check_dir = tempfile.mkdtemp(prefix="liflow_check_")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(check_dir)
    ok = True
    for rel, meta in entries.items():
        with open(os.path.join(check_dir, rel), "rb") as f:
            data = f.read()
        if len(data) != meta["size"] or sha256_bytes(data) != meta["sha256"]:
            ok = False
    shutil.rmtree(check_dir, ignore_errors=True)
    if not ok:
        raise RuntimeError("archive verification failed")
    print(json.dumps(manifest, indent=2))
    print(f"verification: OK ({len(entries)} entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-yaml", default=os.path.join(
        TASK, "configs", "liflow_universal_inference.yaml"))
    ap.add_argument("--propagator", required=True)
    ap.add_argument("--corrector", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    build(args.config_yaml, args.propagator, args.corrector, args.out_dir)


if __name__ == "__main__":
    main()
