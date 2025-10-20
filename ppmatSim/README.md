
# Simulation Tasks
This section explains how to run Molecular Dynamics (MD) or structure optimization tasks using the ASE interface. These tasks are executed via the ppmatSim/main.py script.

You can override any YAML parameter directly from the command line using parameter=value, or write your own YAML configuration.

The Hydra output directory automatically includes the job name and timestamp, so results are well organized and won't be overwritten.

| Section            | Parameter         | Description                                                    |
| ------------------ | ----------------- | -------------------------------------------------------------- |
| `device`           | `cuda`            | Device for computations (`cpu` or `cuda`)                      |
| `model/load_model` | `model_name`      | Pre-trained model name                                         |
|                    | `config_path`     | Path to YAML config (used with checkpoint)                     |
|                    | `checkpoint_path` | Path to model checkpoint (*.pdparams)                          |
| `system`           | `load_system`     | Load initial system from file                                  |
|                    | `ase_create`      | Generate system using ASE                                      |
| `task`             | `md`, `opt`       | Task type: `md` for molecular dynamics, `opt` for optimization |
| `calculator`       | `ase`             | Backend interface (ASE in this case)                           |


## 1. Running Molecular Dynamics (MD) Simulation (with ASE backend)

The MD simulation is implemented in the function
ASECalculator.run_md() within ppmat/calculator/ase.py

By default, it uses the ASE Langevin integrator for time evolution:
```bash
dyn = Langevin(
      atoms,
      timestep=timestep * units.fs,
      temperature_K=temperature,
      friction=0.01 / units.fs,
)
```
This setup enables NVT dynamics with stochastic thermalization, suitable for general-purpose molecular simulations.

Example Usage:
```bash
# Option A: Use a pre-trained model by name
python ppmatSim/main.py --config-name md_ase Model.model_name='chgnet_mptrj'

python ppmatSim/main.py --config-name md_ase Model.model_name='mattersim_1M'

# Option B: Use a custom config and checkpoint
python ppmatSim/main.py --config-name md_ase Model.config_path='your config path(*.yaml)' Model.checkpoint_path='your checkpoint path(*.pdparams)'

python ppmatSim/main.py --config-name md_ase Model.config_path='your config path(*.yaml)' Model.checkpoint_path='your checkpoint path(*.pdparams)'
```

After the simulation, the generated trajectory (.traj) file can be easily converted to an .xyz file for visualization:
```bash
# Trajectory file (.traj) can be converted to XYZ file:
ase convert <trajectory_file>.traj <output_file>.xyz
```


Customization Example:

Users can easily replace the default integrator with other ASE MD engines, such as IsotropicMTKNPT for NPT ensemble simulations or custom pre-relaxation steps.For example:
```bash
from ase.optimize import QuasiNewton
from ase.geometry.analysis import Analysis
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.analysis import DiffusionCoefficient

# Quick relaxation of the initial structure
qn = QuasiNewton(atoms)
qn.run(fmax=0.001, steps=10)

# Initialize velocities and remove net translation/rotation
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
ZeroRotation(atoms)

# Run MD with NPT ensemble
dyn = IsotropicMTKNPT(
    atoms=atoms,
    timestep=timestep * units.fs,
    temperature_K=temperature,
    pressure_au=1 * units.bar,
    tdamp=100,
    pdamp=1000,
    logfile=log_file,
)

```
This flexibility allows users to test different thermodynamic ensembles or thermostats/barostats directly within the ASE interface.

> Coming soon: Examples for running MD with the **LAMMPS backend** will be provided in the next release.


## 2. Running Structure Optimization (with ASE backend)

Structure optimization is supported through ASE’s built-in optimizers.
In the source code, the optimizer class and filter type can be specified via configuration.

The filter (e.g., FrechetCellFilter) is used to apply optimization constraints on both atomic positions and cell parameters, ensuring stable relaxation for periodic systems.

Available Optimizers
```bash
FIRE
BFGS
LBFGS
MDMin
GPMin
LBFGSLineSearch
BFGSLineSearch
```

Default Settings
```bash
optimizer: "LBFGS"
filter: "FrechetCellFilter"
```

Example Usage:
```bash
# Option A: Use a pre-trained model by name
python ppmatSim/main.py --config-name optimizer_ase Model.model_name='chgnet_mptrj'

python ppmatSim/main.py --config-name optimizer_ase Model.model_name='mattersim_1M'

# Option B: Use a custom config and checkpoint
python ppmatSim/main.py --config-name optimizer_ase Model.config_path='your config path(*.yaml)' Model.checkpoint_path='your checkpoint path(*.pdparams)'

python ppmatSim/main.py --config-name optimizer_ase Model.config_path='your config path(*.yaml)' Model.checkpoint_path='your checkpoint path(*.pdparams)'
```
