# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

import argparse
import json

import numpy as np

from ppmat.metrics.liflow_metrics import relative_error


def evaluate(reference, prediction, threshold=5.0):
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    metrics = {
        "final_step": relative_error(
            np.linalg.norm(prediction[-1]), np.linalg.norm(reference[-1])
        ),
        "msd_frame": relative_error(
            np.mean((prediction - prediction[0]) ** 2),
            np.mean((reference - reference[0]) ** 2),
        ),
    }
    if max(metrics.values()) > threshold:
        raise SystemExit(f"sampling metric exceeds {threshold}%: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prediction", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(np.load(args.reference), np.load(args.prediction)), indent=2))
