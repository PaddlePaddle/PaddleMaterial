from pymatgen.core import Structure

from ppmat.models.chgnet.model import StructOptimizer
from ppmat.models.chgnet.model.model import CHGNet

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
