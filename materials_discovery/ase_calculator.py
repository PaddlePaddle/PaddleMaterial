import logging

import numpy as np
from ase import Atoms
from ase import units
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import all_changes
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.velocitydistribution import Stationary
from ase.md.verlet import VelocityVerlet
from gemnet.training.data_container import DataContainer


class Molecule(DataContainer):
    """
    Implements the DataContainer but for a single molecule. Requires custom
    init method.
    """

    def __init__(self, R, Z, cutoff, int_cutoff, triplets_only=False):
        self.index_keys = [
            "batch_seg",
            "id_undir",
            "id_swap",
            "id_c",
            "id_a",
            "id3_expand_ba",
            "id3_reduce_ca",
            "Kidx3",
        ]
        if not triplets_only:
            self.index_keys += [
                "id4_int_b",
                "id4_int_a",
                "id4_reduce_ca",
                "id4_expand_db",
                "id4_reduce_cab",
                "id4_expand_abd",
                "Kidx4",
                "id4_reduce_intm_ca",
                "id4_expand_intm_db",
                "id4_reduce_intm_ab",
                "id4_expand_intm_ab",
            ]
        self.triplets_only = triplets_only
        self.cutoff = cutoff
        self.int_cutoff = int_cutoff
        self.keys = ["N", "Z", "R", "F", "E"]
        assert tuple(R.shape) == (len(Z), 3)
        self.R = R
        self.Z = Z
        self.N = np.array([len(Z)], dtype=np.int32)
        self.E = np.zeros(1, dtype=np.float32).reshape(1, 1)
        self.F = np.zeros((len(Z), 3), dtype=np.float32)
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        self.addID = False
        self.dtypes, dtypes2 = self.get_dtypes()
        self.dtypes.update(dtypes2)
        self.device = "cpu"

    def get(self):
        """
        Get the molecule representation in the expected format for the GemNet
        model.
        """
        data = self.__getitem__(0)
        for var in ["E", "F"]:
            data.pop(var)
        for key in data:
            data[key] = data[key].to(self.device)
        return data

    def update(self, R):
        """
        Update the position of the atoms.
        Graph representation of the molecule might change if the atom positions
        are updated.

        Parameters
        ----------
        R: torch.Tensor (nAtoms, 3)
            Positions of the atoms in A°.
        """
        assert tuple(self.R.shape) == tuple(R.shape)
        self.R = R

    def to(self, device):
        """
        Changes the device of the returned tensors in the .get() method.
        """
        self.device = device


class GNNCalculator(Calculator):
    """
    A custom ase calculator that computes energy and forces acting on atoms of
    a molecule using GNNs,
    e.g. GemNet.

    Parameters
    ----------
    molecule
        Captures data of all atoms. Contains indices etc.
    model
        The trained GemNet model.
    atoms: ase.Atoms
        ASE atoms instance.

    restart: str
        Prefix for restart file.  May contain a directory. Default is None:
        don't restart.
    label: str
        Name used for all files.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        molecule,
        model,
        atoms=None,
        restart=None,
        add_atom_energies=False,
        label="gemnet_calc",
        **kwargs,
    ):
        super().__init__(restart=restart, label=label, atoms=atoms, **kwargs)
        self.molecule = molecule
        self.model = model
        self.add_atom_energies = add_atom_energies
        self.atom_energies = {
            (1): -13.641404161,
            (6): -1027.592489146,
            (7): -1484.274819088,
            (8): -2039.734879322,
            (16): -10828.707468187,
            (17): -12516.444619523,
        }

    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        self.molecule.update(R=atoms.positions)
        inputs = self.molecule.get()
        energy, forces = self.model.predict(inputs)
        energy = float(energy)
        if self.add_atom_energies:
            energy += np.sum([self.atom_energies[z] for z in atoms.numbers])
        self.results["energy"] = energy
        self.results["forces"] = forces.numpy()


class MDSimulator:
    """
    Runs a MD simulation on the Atoms object created from data and perform MD
    simulation for max_steps

    Parameters
    ----------
    molecule
        Captures data of all atoms.
    model
        The trained GemNet model.
    dynamics: str
        Name of the MD integrator. Implemented: 'langevin' or 'verlet'.
    max_steps: int
        Maximum number of simulation steps.
    time: float
        Integration time step for Newton's law in femtoseconds.
    temperature: float
        The temperature in Kelvin.
    langevin_friction: float
        Only used when dynamics are 'langevin'. A friction coefficient,
        typically 1e-4 to 1e-2.
    interval: int
        Write only every <interval> time step to trajectory file.
    traj_path: str
        Path of the file where to save the calculated trajectory.
    vel: N-array, default=None
        If set, then atoms have been initialized with these velocties.
    logfile: str
        File name or open file, where to log md simulation. “-” refers to
        standard output.
    """

    def __init__(
        self,
        molecule,
        model,
        dynamics: str = "langevin",
        max_steps: int = 100,
        time: float = 0.5,
        temperature: float = 300,
        langevin_friction: float = 0.002,
        interval: int = 10,
        traj_path="md_sim.traj",
        vel=None,
        logfile="-",
    ):
        self.max_steps = max_steps
        atoms = Atoms(positions=molecule.R, numbers=molecule.Z)
        atoms.calc = GNNCalculator(molecule, model=model, atoms=atoms)
        if vel is not None:
            atoms.set_velocities(vel)
        else:
            MaxwellBoltzmannDistribution(atoms, temp=temperature * units.kB)
            Stationary(atoms)
        self.dyn = None
        if dynamics.lower() == "verlet":
            logging.info("Selected MD integrator: Verlet")
            self.dyn = VelocityVerlet(atoms, timestep=time * units.fs)
        elif dynamics.lower() == "langevin":
            logging.info("Selected MD integrator: Langevin")
            self.dyn = Langevin(
                atoms,
                timestep=time * units.fs,
                temperature=temperature * units.kB,
                friction=langevin_friction,
            )
        else:
            raise UserWarning(
                f"""Unkown MD integrator. I only know 'verlet' and 'langevin'"
                    "but {dynamics} was given."""
            )
        logging.info(f"Save trajectory to {traj_path}")
        self.traj = Trajectory(traj_path, "w", atoms)
        self.dyn.attach(self.traj.write, interval=interval)
        self.dyn.attach(
            MDLogger(self.dyn, atoms, logfile, peratom=False, mode="a"),
            interval=interval,
        )

    def run(self):
        self.dyn.run(self.max_steps)
        self.traj.close()
