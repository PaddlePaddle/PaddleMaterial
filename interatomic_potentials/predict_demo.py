from ase.io.trajectory import Trajectory
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import interatomic_potentials.eager_comp_setting as eager_comp_setting
from ppmat.models.chgnet.model import StructOptimizer
from ppmat.models.chgnet.model.dynamics import MolecularDynamics
from ppmat.models.chgnet.model.model import CHGNet
from ppmat.models.chgnet.utils.vasp_utils import solve_charge_by_mag

eager_comp_setting.setting_eager_mode(enable=True)

chgnet = CHGNet.load()
structure = Structure.from_file("interatomic_potentials/mp-18767-LiMnO2.cif")
prediction = chgnet.predict_structure(structure)

for key, unit in [
    ("energy", "eV/atom"),
    ("forces", "eV/A"),
    ("stress", "GPa"),
    ("magmom", "mu_B"),
]:
    print(f"CHGNet-predicted {key} ({unit}):\n{prediction[key[0]]}\n")


relaxer = StructOptimizer()

# Perturb the structure
structure.perturb(0.8)

# Relax the perturbed structure
result = relaxer.relax(structure, verbose=True)

print("Relaxed structure:\n")
print(result["final_structure"])

print(result["trajectory"].energies)


md = MolecularDynamics(
    atoms=structure,
    model=chgnet,
    ensemble="nvt",
    temperature=1000,  # in k
    timestep=2,  # in fs
    trajectory="md_out.traj",
    logfile="md_out.log",
    loginterval=100,
)
md.run(50)  # run a 0.1 ps MD simulation


traj = Trajectory("md_out.traj")
mag = traj[-1].get_magnetic_moments()

# get the non-charge-decorated structure
structure = AseAtomsAdaptor.get_structure(traj[-1])
print(structure)

# get the charge-decorated structure
struct_with_chg = solve_charge_by_mag(structure)
print(struct_with_chg)
