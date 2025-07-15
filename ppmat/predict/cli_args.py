"""
python main.py predict --file_path ./property_prediction/example_data/cifs/
"""

import argparse


def add_common_args(p):
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model to use for prediction.",
    )
    p.add_argument(
        "--weights_name",
        type=str,
        default=None,
        help="Name of the model weight file, e.g., best.pdparams or latest.pdparams.",
    )
    p.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the configuration YAML file.",
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the trained model checkpoint (for inference).",
    )
    p.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="Path to a CIF file or a directory containing structure files to predict.",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the prediction results (CSV).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device name, e.g., gpu, cpu.",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="PPMaterial Prediction CLI")

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available sub-commands"
    )

    # General prediction
    pred_parser = subparsers.add_parser("predict", help="Predict properties")
    add_common_args(pred_parser)

    # MD Simulation
    ase_md_parser = subparsers.add_parser("ase_md", help="Using ASE runs MD simulation")
    add_common_args(ase_md_parser)
    ase_md_parser.add_argument(
        "--ase_calc", action="store_true", help="Use ASE calculator (default: False)."
    )
    ase_md_parser.add_argument(
        "--temperature", type=float, default=300, help="Temperature in Kelvin."
    )
    ase_md_parser.add_argument(
        "--timestep", type=float, default=0.1, help="Timestep for MD simulation in fs."
    )
    ase_md_parser.add_argument(
        "--steps", type=int, default=1000, help="Number of MD steps."
    )
    ase_md_parser.add_argument(
        "--interval", type=int, default=1, help="Interval to save trajectory."
    )

    # Structure Optimization
    opt_parser = subparsers.add_parser("ase_opt", help="Using ASE optimizes structure")
    add_common_args(opt_parser)
    opt_parser.add_argument(
        "--ase_calc", action="store_true", help="Use ASE calculator (default: False)."
    )
    opt_parser.add_argument(
        "--optimizer",
        type=str,
        default="LBFGS",
        choices=[
            "FIRE",
            "BFGS",
            "LBFGS",
            "MDMin",
            "GPMin",
            "LBFGSLineSearch",
            "BFGSLineSearch",
        ],
        help="Optimizer name.",
    )
    opt_parser.add_argument(
        "--filter",
        type=str,
        default="FrechetCellFilter",
        choices=[
            "none",
            "Filter",
            "StrainFilter",
            "UnitCellFilter",
            "FrechetCellFilter",
            "ExpCellFilter",
        ],
        help="Filter name.",
    )
    opt_parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Maximum force tolerance.",
    )
    opt_parser.add_argument("--steps", type=int, default=100, help="Number of steps.")

    return parser.parse_args()
