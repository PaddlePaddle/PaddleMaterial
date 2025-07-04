import chemparse
import numpy as np
import paddle

from ppmat.datasets.collate_fn import Data
from ppmat.datasets.num_atom_dists import NUM_ATOM_DIST
from ppmat.utils import DEFAULT_ELEMENTS
from ppmat.utils import paddle_aux  # noqa


class GenDataset(paddle.io.Dataset):
    def __init__(
        self,
        total_num,
        formula=None,
        dist_name="mp_20",
        prop_names=None,
        prop_values=None,
    ):
        super().__init__()

        self.total_num = total_num
        self.formula = formula
        if formula is not None:
            self.chem_list = self.get_structure(formula)
        else:
            self.chem_list = None

            assert dist_name in NUM_ATOM_DIST
            distribution = NUM_ATOM_DIST.get(dist_name)
            self.distribution = distribution
            self.num_atoms = np.random.choice(
                len(self.distribution), total_num, p=self.distribution
            )
        self.prop_names = prop_names
        self.prop_values = prop_values
        if prop_names is not None and prop_values is not None:
            assert len(prop_names) == len(prop_values)
            self.prop_flag = True
        else:
            self.prop_flag = False

    def get_structure(self, formula):
        composition = chemparse.parse_formula(formula)
        chem_list = []
        for elem in composition:
            num_int = int(composition[elem])
            chem_list.extend([DEFAULT_ELEMENTS.index(elem)] * num_int)
        return chem_list

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        data = {}
        if self.chem_list is None:
            num_atom = self.num_atoms[index]
            data["structure_array"] = Data(
                {
                    "num_atoms": np.array([num_atom]),
                }
            )
        else:
            data["structure_array"] = Data(
                {
                    "num_atoms": np.array([len(self.chem_list)]),
                    "atom_types": np.array(self.chem_list),
                }
            )
        data["num_atoms"] = data["structure_array"].num_atoms

        if self.prop_flag:
            for prop_name, prop_value in zip(self.prop_names, self.prop_values):
                data[prop_name] = np.array([prop_value]).astype("float32")
        return data
