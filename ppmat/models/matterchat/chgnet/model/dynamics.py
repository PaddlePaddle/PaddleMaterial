# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import contextlib
import io
import pickle
import sys
from typing import TYPE_CHECKING

import numpy as np
import paddle
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes, all_properties
from ase.constraints import ExpCellFilter
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from pymatgen.analysis.eos import BirchMurnaghan
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.models.matterchat.chgnet.model.model import CHGNet
from ppmat.utils import logger

if TYPE_CHECKING:
    from ase.io import Trajectory
    from ase.optimize.optimize import Optimizer

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "BFGSLineSearch": BFGSLineSearch,
}


class CHGNetCalculator(Calculator):
    """CHGNet Calculator for ASE applications."""

    implemented_properties = ["energy", "forces", "stress", "magmoms"]

    def __init__(
        self,
        model: CHGNet | None = None,
        use_device: str | None = None,
        stress_weight: float | None = 1 / 160.21766208,
        **kwargs,
    ) -> None:
        """Provide a CHGNet instance to calculate various atomic properties using ASE."""
        super().__init__(**kwargs)

        if use_device == "mps":
            raise NotImplementedError("mps is not supported yet")
        self.device = use_device or (
            "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        )

        if model is None:
            model = CHGNet.load()
        paddle.set_device(self.device)
        self.model = model
        self.stress_weight = stress_weight
        logger.message(f"CHGNet will run on {self.device}")

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ) -> None:
        """Calculate various properties of the atoms using CHGNet."""
        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        structure = AseAtomsAdaptor.get_structure(atoms)
        graph = self.model.graph_converter(structure)
        model_prediction = self.model.predict_graph(
            graph.to(self.device), task="efsm"
        )

        factor = (
            1
            if not self.model.is_intensive
            else structure.composition.num_atoms
        )
        self.results.update(
            energy=model_prediction["e"] * factor,
            forces=model_prediction["f"],
            free_energy=model_prediction["e"] * factor,
            magmoms=model_prediction["m"],
            stress=model_prediction["s"] * self.stress_weight,
        )


class StructOptimizer:
    """Wrapper class for structural relaxation."""

    def __init__(
        self,
        model: CHGNet | None = None,
        optimizer_class: Optimizer | str | None = "FIRE",
        use_device: str | None = None,
        stress_weight: float = 1 / 160.21766208,
    ) -> None:
        """Provide a trained CHGNet model and an optimizer to relax crystal structures."""
        if isinstance(optimizer_class, str):
            if optimizer_class in OPTIMIZERS:
                optimizer_class = OPTIMIZERS[optimizer_class]
            else:
                raise ValueError(
                    f"Optimizer instance not found. Select one from {list(OPTIMIZERS)}"
                )

        self.optimizer_class: Optimizer = optimizer_class
        self.calculator = CHGNetCalculator(
            model=model, stress_weight=stress_weight, use_device=use_device
        )

    def relax(
        self,
        atoms: Structure | Atoms,
        fmax: float | None = 0.1,
        steps: int | None = 500,
        relax_cell: bool | None = True,
        save_path: str | None = None,
        trajectory_save_interval: int | None = 1,
        verbose: bool = True,
        **kwargs,
    ) -> dict[str, Structure | TrajectoryObserver]:
        """Relax the Structure/Atoms until maximum force is smaller than fmax."""
        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor.get_atoms(atoms)

        atoms.calc = self.calculator

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if relax_cell:
                atoms = ExpCellFilter(atoms)
            optimizer = self.optimizer_class(atoms, **kwargs)
            optimizer.attach(obs, interval=trajectory_save_interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()

        if save_path is not None:
            obs.save(save_path)

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)
        for k in struct.site_properties:
            struct.remove_site_property(property_name=k)
        struct.add_site_property(
            "magmom", [float(i) for i in atoms.get_magnetic_moments()]
        )
        return {"final_structure": struct, "trajectory": obs}


class TrajectoryObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a TrajectoryObserver from an Atoms object."""
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.magmoms: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self):
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy."""
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        """Save the trajectory to file."""
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class MolecularDynamics:
    """Molecular dynamics class."""

    def __init__(
        self,
        atoms: Atoms | Structure,
        model: CHGNet | None = None,
        ensemble: str = "nvt",
        temperature: int = 300,
        timestep: float = 2.0,
        pressure: float = 1.01325 * units.bar,
        taut: float | None = None,
        taup: float | None = None,
        compressibility_au: float | None = None,
        trajectory: str | Trajectory | None = None,
        logfile: str | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        use_device: str | None = None,
    ) -> None:
        """Initialize the MD class."""
        if isinstance(atoms, (Structure, Molecule)):
            atoms = AseAtomsAdaptor.get_atoms(atoms)

        self.atoms = atoms
        self.atoms.calc = CHGNetCalculator(model, use_device=use_device)

        if taut is None:
            taut = 100 * timestep * units.fs
        if taup is None:
            taup = 1000 * timestep * units.fs

        if ensemble.lower() == "nvt":
            self.dyn = NVTBerendsen(
                atoms=self.atoms,
                timestep=timestep * units.fs,
                temperature_K=temperature,
                taut=taut,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )
        else:
            if compressibility_au is None:
                eos = EquationOfState(
                    model=model,
                    use_device=use_device,
                )
                eos.fit(atoms=atoms, steps=500, fmax=0.1)
                compressibility_au = eos.get_compressibility(unit="A^3/eV")
                logger.message(
                    f"Done compressibility calculation: "
                    f"b = {round(compressibility_au, 3)} A^3/eV"
                )

            if ensemble.lower() == "npt":
                self.dyn = Inhomogeneous_NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure,
                    taut=taut,
                    taup=taup,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                )

            elif ensemble.lower() == "npt_berendsen":
                self.dyn = NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure,
                    taut=taut,
                    taup=taup,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )

            else:
                raise ValueError("Ensemble not supported")

        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep

    def run(self, steps: int):
        """Thin wrapper of ase MD run."""
        self.dyn.run(steps)

    def set_atoms(self, atoms: Atoms):
        """Set new atoms to run MD."""
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.calc = calculator


class EquationOfState:
    """Class to calculate equation of state."""

    def __init__(
        self,
        model: CHGNet | None = None,
        optimizer_class: Optimizer | str | None = "FIRE",
        use_device: str | None = None,
        stress_weight: float = 1 / 160.21766208,
    ) -> None:
        """Initialize a structure optimizer object for calculation of bulk modulus."""
        self.relaxer = StructOptimizer(
            model=model,
            optimizer_class=optimizer_class,
            use_device=use_device,
            stress_weight=stress_weight,
        )
        self.fitted = False

    def fit(
        self,
        atoms: Structure | Atoms,
        n_points: int = 11,
        fmax: float | None = 0.1,
        steps: int | None = 500,
        **kwargs,
    ):
        """Relax the Structure/Atoms and fit the Birch-Murnaghan equation of state."""
        if isinstance(atoms, Atoms):
            atoms = AseAtomsAdaptor.get_structure(atoms)
        volumes, energies = [], []
        for i in np.linspace(-0.1, 0.1, n_points):
            structure_strained = atoms.copy()
            structure_strained.apply_strain([i, i, i])
            result = self.relaxer.relax(
                structure_strained,
                relax_cell=False,
                fmax=fmax,
                steps=steps,
                verbose=False,
                **kwargs,
            )
            volumes.append(result["final_structure"].volume)
            energies.append(result["trajectory"].energies[-1])
        self.bm = BirchMurnaghan(volumes=volumes, energies=energies)
        self.bm.fit()
        self.fitted = True

    def get_bulk_mudulus(self, unit: str = "eV/A^3"):
        """Get the bulk modulus from the fitted Birch-Murnaghan equation of state."""
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "eV/A^3":
            return self.bm.b0
        if unit == "GPa":
            return self.bm.b0_GPa
        raise NotImplementedError("unit has to be eV/A^3 or GPa")

    def get_compressibility(self, unit: str = "A^3/eV"):
        """Get the compressibility from the fitted Birch-Murnaghan equation of state."""
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "A^3/eV":
            return 1 / self.bm.b0
        if unit == "GPa^-1":
            return 1 / self.bm.b0_GPa
        if unit in ["Pa^-1", "m^2/N"]:
            return 1 / (self.bm.b0_GPa * 1e9)
        raise NotImplementedError(
            "unit has to be one of A^3/eV, GPa^-1 Pa^-1 or m^2/N"
        )
