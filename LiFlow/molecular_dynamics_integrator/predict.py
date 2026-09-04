# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse

import numpy as np

from ppmat.predictor.integrator_predictor import IntegratorPredictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--weights_name", default=None)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--index_file", default=None)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--flow_steps", type=int, default=None)
    parser.add_argument("--solver", choices=["euler", "heun"], default=None)
    parser.add_argument("--corrector_every", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    predictor = IntegratorPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    result = predictor.from_dataset_sample(
        data_path=args.data_path,
        index_file=args.index_file,
        sample_index=args.sample_index,
    )
    if args.steps is not None:
        trajectory = predictor.run(
            result["positions"],
            steps=args.steps,
            flow_steps=args.flow_steps or 10,
            solver=args.solver or "euler",
            corrector_every=args.corrector_every or 1,
            seed=args.seed if args.seed is not None else 42,
        )
        if args.output:
            np.save(args.output, trajectory)
        print(f"trajectory_shape={trajectory.shape}")
    else:
        velocity = result["velocity"].numpy()
        target = result["target"].numpy()
        mse = float(((velocity - target) ** 2).sum(axis=-1).mean())
        print(f"sample={result['name']} frames={result['frame_start']}->{result['frame_end']} atoms={result['num_atoms']} mse={mse:.8f}")
        print("predicted_velocity:")
        print(velocity)
