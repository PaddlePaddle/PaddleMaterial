#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q ppmat molecular_dynamics_integrator
python -m pytest -q test/test_liflow_layers.py test/test_liflow_model.py test/test_liflow_dataset.py test/test_liflow_predictor.py test/test_liflow_integration.py
