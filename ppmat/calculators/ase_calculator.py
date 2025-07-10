import os
import os.path as osp
from typing import Any
from typing import List
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress
from ppmat.calculators.base_calculator import PPMatPredictor
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from ppmat.utils import logger

CHARGE_RANGE = [-100, 100]
SPIN_RANGE = [0, 100]
DEFAULT_CHARGE = 0
DEFAULT_SPIN_OMOL = 1
DEFAULT_SPIN = 0


class PPMatCalculator(Calculator):
    def __init__(
        self,
        predictor: "PPMatPredictor",
        **kwargs: Any,
    ):
        """
        Initialize the PPMatCalculator from a model PPMatPredictor

        Args:
            predict_unit (PPMatPredictor): A pretrained PPMatPredictor.
        Notes:
            - For models that require total charge and spin multiplicity
                `charge` and `spin` (corresponding to `spin_multiplicity`) are
                    pulled from `atoms.info` during calculations.
                - `charge` must be an integer representing the total charge
                    on the system and can range from -100 to 100.
                - `spin` must be an integer representing the spin multiplicity
                    and can range from 0 to 100.
                - If `charge` or `spin` are not set in `atoms.info`,
                    they will default to charge=`0` and spin=`1`.
        """

        super().__init__(**kwargs)

        label_names = predictor.config["Global"]["label_names"]
        model_cls = predictor.config["Model"].get("__class_name__")
        if model_cls == "CHGNet":
            if "energy_per_atom" in label_names:
                logger.warning(
                    "'energy_per_atom' found in prediction. "
                    "Please ensure this corresponds to total energy as expected."
                )
            label_names = ["forces" if x == "force" else x for x in label_names]
            if "energy_per_atom" in label_names and "energy" not in label_names:
                label_names.append("energy")
        else:
            raise NotImplementedError(
                f"Model {model_cls} not supported. "
                f"Check that predicted properties (forces, energy, stress) "
                f"match ASE requirements. Add handling if needed."
            )

        self.implemented_properties = label_names
        
        self.predictor = predictor

    def from_cif_file(
        self,
        file_path: Optional[str] = None,
    ):
        # TODO: Currently only support cif format files
        if osp.isdir(file_path):
            cif_files = [
                osp.join(file_path, f)
                for f in os.listdir(file_path)
                if f.endswith(".cif")
            ]
        else:
            if file_path.endswith(".cif"):
                cif_files = [file_path]
            else:
                raise ValueError(f"Invalid CIF file path: {file_path}")
        # Read raw file and convert to ASE Atom
        structures = []
        for cif_file in tqdm(cif_files):
            structure = Structure.from_file(cif_file)
            graph = self.predictor.graph_converter(structure)
            # covert pymatgen Structure to ASE Atom
            atom = AseAtomsAdaptor().get_atoms(structure)
            # attach graph information to ASE Atom
            atom.info["graph"] = graph
            structures.append(atom)
        logger.info("Successfully covert all structures to ASE Atoms.")
        return structures

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list:
        """
        Check for any system changes since the last calculation.

        Args:
            atoms (ase.Atoms): The atomic structure to check.
            tol (float): Tolerance for detecting changes.

        Returns:
            list: A list of changes detected in the system.
        """
        state = super().check_state(atoms, tol=tol)
        if (not state) and (self.atoms.info != atoms.info):
            state.append("info")
        return state

    def calculate(
        self,
        atoms: Atoms,
        properties: List[str],
        system_changes: List[str],
    ) -> None:
        """
        Perform the calculation for the given atomic structure.

        Args:
            atoms (Atoms): The atomic structure to calculate properties for.
            properties (list[str]): The list of properties to calculate.
            system_changes (list[str]): The list of changes in the system.

        Notes:
            - `charge` must be an integer representing the total charge
                on the system and can range from -100 to 100.
            - `spin` must be an integer representing the spin multiplicity
                and can range from 0 to 100.
            - If `charge` or `spin` are not set in `atoms.info`,
                they will default to `0`.
            - The `free_energy` is simply a copy of the `energy`
                and is not the actual electronic free energy.
              It is only set for ASE routines/optimizers that are hard-coded to use
                this rather than the `energy` key.
        """

        # Our calculators won't work if natoms=0
        if len(atoms) == 0:
            raise ValueError("Atoms object has no atoms inside.")

        # Check if the atoms object has periodic boundary conditions (PBC) set correctly
        self._check_atoms_pbc(atoms)

        # Validate that charge/spin are set correctly, or default to 0 otherwise
        self._validate_charge_and_spin(atoms)

        # Standard call to check system_changes etc
        Calculator.calculate(self, atoms, properties, system_changes)

        if len(atoms) == 1 and sum(atoms.pbc) == 0:
            pass  # self.results = self._get_single_atom_energies(atoms)
        else:
            # Predict
            graph = atoms.info["graph"]
            pred = self.predictor.model.predict(graph)
            pred = self.predictor.post_process(pred)

            # Collect the results into self.results
            self.results = {}

            # energy
            if "energy" in self.implemented_properties and "energy" in pred:
                energy = float(pred["energy"].squeeze())
            elif (
                "energy_per_atom" in self.implemented_properties
                and "energy_per_atom" in pred
            ):
                energy = float(pred["energy_per_atom"].squeeze())
            else:
                raise KeyError(
                    "Neither 'energy' nor 'energy_per_atom' found in prediction."
                )
            self.results["energy"] = self.results[
                "free_energy"
            ] = energy  # Free energy is a copy of energy

            # forces, stress
            name_map = {
                "forces": "force",
            }
            for calc_key in self.implemented_properties:
                pred_key = name_map.get(calc_key, calc_key)
                if calc_key in ("force", "forces"):
                    self.results["forces"] = pred[pred_key]
                elif calc_key == "stress":
                    stress = pred[pred_key]
                    stress_voigt = full_3x3_to_voigt_6_stress(stress)
                    self.results["stress"] = stress_voigt

    def _check_atoms_pbc(self, atoms) -> None:
        """
        Check for invalid PBC conditions

        Args:
            atoms (ase.Atoms): The atomic structure to check.
        """
        if np.all(atoms.pbc) and np.allclose(atoms.cell, 0):
            raise AllZeroUnitCellError
        if np.any(atoms.pbc) and not np.all(atoms.pbc):
            raise MixedPBCError

    def _validate_charge_and_spin(self, atoms: Atoms) -> None:
        """
        Validate and set default values for charge and spin.

        Args:
            atoms (Atoms): The atomic structure containing charge and spin information.
        """

        if "charge" not in atoms.info:
            atoms.info["charge"] = DEFAULT_CHARGE
            logger.warning(
                "Defaulting to charge=0. "
                "Ensure charge is an integer representing the total charge "
                "on the system and is within the range -100 to 100."
            )

        if "spin" not in atoms.info:
            atoms.info["spin"] = DEFAULT_SPIN
            logger.warning(
                "Defaulting to spin=1. "
                "Ensure spin is an integer representing "
                "the spin multiplicity from 0 to 100."
            )

        # Validate charge
        charge = atoms.info["charge"]
        if not isinstance(charge, int):
            raise TypeError(
                f"Invalid type for charge: {type(charge)}. "
                f"Charge must be an integer representing "
                f"the total charge on the system."
            )
        if not (CHARGE_RANGE[0] <= charge <= CHARGE_RANGE[1]):
            raise ValueError(
                f"Invalid value for charge: {charge}. "
                f"Charge must be within the range "
                f"{CHARGE_RANGE[0]} to {CHARGE_RANGE[1]}."
            )

        # Validate spin
        spin = atoms.info["spin"]
        if not isinstance(spin, int):
            raise TypeError(
                f"Invalid type for spin: {type(spin)}. "
                f"Spin must be an integer representing "
                f"the spin multiplicity."
            )
        if not (SPIN_RANGE[0] <= spin <= SPIN_RANGE[1]):
            raise ValueError(
                f"Invalid value for spin: {spin}. "
                f"Spin must be within the range "
                f"{SPIN_RANGE[0]} to {SPIN_RANGE[1]}."
            )


class MixedPBCError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Attempted to guess PBC for an atoms object, "
        "but the atoms object has PBC set to True for somedimensions but not others. "
        "Please ensure that the atoms object has PBC set to True for all dimensions.",
    ):
        self.message = message
        super().__init__(self.message)


class AllZeroUnitCellError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Atoms object claims to have PBC set, "
        "but the unit cell is identically 0. Please ensure that the atoms "
        "object has a non-zero unit cell.",
    ):
        self.message = message
        super().__init__(self.message)
