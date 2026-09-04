#!/usr/bin/env bash
set -euo pipefail
python molecular_dynamics_integrator/evaluate.py --reference "$1" --prediction "$2"
