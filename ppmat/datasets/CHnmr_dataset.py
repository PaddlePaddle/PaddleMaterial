import os.path as osp
import pathlib
import pickle
from typing import Dict
from typing import Literal
from typing import Optional

import numpy as np
import paddle
import pandas as pd
from paddle.io import DataLoader
from paddle.io import Dataset
from paddle.vision.transforms import Compose
from rdkit import Chem

from ppmat.datasets.ext_rdkit import build_molecule_with_partial_charges
from ppmat.datasets.ext_rdkit import compute_molecular_metrics
from ppmat.datasets.ext_rdkit import mol2smiles
from ppmat.datasets.structure_converter import Mol2Graph
from ppmat.datasets.utils import build_molecules_from_smiles
from ppmat.datasets.utils import numericalize_text
from ppmat.utils import DEFAULT_ELEMENTS
from ppmat.utils import logger


class CHnmrDataset(Dataset):
    """
    datasets for Spectrum Graph Molecules
    """

    def __init__(
        self,
        path: str,
        remove_h=False,
        target_prop=None,
        transform=None,
        num_cpus: Optional[int] = None,
        element_types: Literal["DEFAULT_ELEMENTS"] = "DEFAULT_ELEMENTS",
        converter_cfg: Dict = None,
        filter_key: Optional[str] = None,
        cache: bool = True,
        **kwargs,
    ):
        self.path = path
        self.remove_h = remove_h
        self.target_prop = target_prop
        self.transform = transform
        self.num_cpus = num_cpus
        self.filter_key = filter_key
        self.converter_cfg = converter_cfg
        self.cache = cache

        if cache:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "cached settings match your current settings."
            )

        self.csv_data = self.read_csv(path)
        self.num_samples = len(self.csv_data)
        if element_types.upper() == "DEFAULT_ELEMENTS":
            self.element_types = DEFAULT_ELEMENTS
        else:
            logger.warning("This datasets use customized element types")

        # when cache is True, load cached mols from cache file
        cache_path = osp.join(path.rsplit(".", 1)[0] + "_strucs.pkl")
        if self.cache and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.mols = pickle.load(f)
            logger.info(f"Load {len(self.mols)} cached mols from {cache_path}")
        else:
            # build mols from smiles
            self.mols = build_molecules_from_smiles(
                self.csv_data["smiles"], self.remove_h
            )
            logger.info(f"Build {len(self.mols)} molecules")
            if self.cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.mols, f)
                logger.info(f"Save {len(self.mols)} built moleculs to {cache_path}")

        # build graphs from mols
        if converter_cfg is not None:
            # load cached graphs from cache file
            graph_method = converter_cfg["method"]
            cache_path = osp.join(
                path.rsplit(".", 1)[0] + f"_{graph_method}_graphs.pkl"
            )
            if self.cache and osp.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.graphs = pickle.load(f)
                logger.info(f"Load {len(self.graphs)} cached graphs from {cache_path}")
                assert len(self.graphs) == len(self.mols)
            else:
                # build graphs from molecules
                self.converter = Mol2Graph(**self.converter_cfg)
                self.graphs = self.converter(self.mols)
                logger.info(f"Convert {len(self.graphs)} molecules into graphs")
                if self.cache:
                    with open(cache_path, "wb") as f:
                        pickle.dump(self.graphs, f)
                    logger.info(
                        f"Save {len(self.graphs)} converted graphs to {cache_path}"
                    )
        else:
            self.graphs = None

    def read_csv(self, path):
        data = pd.read_csv(path)
        logger.info(f"Read {len(data)} molecules from {path}")
        data = {key: data[key].tolist() for key in data if "Unnamed" not in key}
        return data

    def __getitem__(self, idx):
        data = {}
        if self.graphs is not None:
            # Obtain the graph from the cache, as this data is frequently utilized
            # for training property prediction models.
            data["graph"] = self.graphs[idx]
        else:
            mol = self.mols[idx]
            data["structure_array"] = self.get_mol_array(mol)

        if "nuclear_magnetic_resonance_spectrum" in self.csv_data:
            data["nuclear_magnetic_resonance_spectrum"] = np.array(
                [self.csv_data["nuclear_magnetic_resonance_spectrum"][idx]]
            ).astype("float32")
        if "atom_account" in self.csv_data:
            data["atom_account"] = np.array(
                [self.csv_data["atom_account"][idx]]
            ).astype("float32")

        data = self.transforms(data) if self.transforms is not None else data
        return data

    def get_mol_array(self, mol):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = {}
        if self.graphs is not None:
            # Obtain the graph from the cache, as this data is frequently utilized
            # for training property prediction models.
            data["graph"] = self.graphs[idx]
        else:
            structure = self.structures[idx]
            data["structure_array"] = self.get_structure_array(structure)
        data["id"] = self.csv_data["material_id"][idx]

        data = self.transforms(data) if self.transforms is not None else data
        return data


class CHnmrDatasetInfos:
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.name = "CHnmr"
        self.atom_encoder = (
            {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
            if not self.remove_h
            else {
                "C": 0,
                "N": 1,
                "O": 2,
                "F": 3,
                "P": 4,
                "S": 5,
                "Cl": 6,
                "Br": 7,
                "I": 8,
            }
        )
        self.atom_decoder = list(self.atom_encoder.keys())
        self.num_atom_types = len(self.atom_encoder)
        self.valencies = (
            [1, 4, 3, 2, 1] if not self.remove_h else [4, 3, 2, 1, 3, 2, 1, 1, 1]
        )
        self.max_n_nodes = 29 if not self.remove_h else 15
        self.max_weight = 390 if not self.remove_h else 564
        self.atom_weights = (
            {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
            if not self.remove_h
            else {
                0: 12,
                1: 14,
                2: 16,
                3: 19,
                4: 30.97,
                5: 32.07,
                6: 35.45,
                7: 79.9,
                8: 126.9,
            }
        )
        self.n_nodes = (
            paddle.to_tensor(
                [
                    0,
                    0,
                    0,
                    1.5287e-05,
                    3.0574e-05,
                    3.8217e-05,
                    9.1721e-05,
                    0.00015287,
                    0.00049682,
                    0.0013147,
                    0.0036918,
                    0.0080486,
                    0.016732,
                    0.03078,
                    0.051654,
                    0.078085,
                    0.10566,
                    0.1297,
                    0.13332,
                    0.1387,
                    0.094802,
                    0.10063,
                    0.033845,
                    0.048628,
                    0.0054421,
                    0.014698,
                    0.00045096,
                    0.0027211,
                    0.0,
                    0.00026752,
                ]
            )
            if not self.remove_h
            else paddle.to_tensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.000657983182463795,
                    0.0034172674641013145,
                    0.009784846566617489,
                    0.019774870947003365,
                    0.04433957487344742,
                    0.07253380119800568,
                    0.10895635187625885,
                    0.14755095541477203,
                    0.17605648934841156,
                    0.19964483380317688,
                    0.21728302538394928,
                ]
            )
        )
        self.node_types = (
            paddle.to_tensor([0.5122, 0.3526, 0.0562, 0.0777, 0.0013])
            if not self.remove_h
            else paddle.to_tensor(
                [
                    0.7162184715270996,
                    0.09598348289728165,
                    0.12478094547986984,
                    0.01828921213746071,
                    0.0004915347089990973,
                    0.014545895159244537,
                    0.01616295613348484,
                    0.011324135586619377,
                    0.002203370677307248,
                ]
            )
        )
        self.edge_types = (
            paddle.to_tensor([0.88162, 0.11062, 0.0059875, 0.0017758, 0])
            if not self.remove_h
            else paddle.to_tensor(
                [
                    0.8293983340263367,
                    0.09064729511737823,
                    0.011958839371800423,
                    0.0011387828271836042,
                    0.0668567642569542,
                ]
            )
        )
        self.valency_distribution = paddle.zeros(3 * self.max_n_nodes - 2)
        if recompute_statistics:
            self.n_nodes = datamodule.node_counts()
            self.node_types = datamodule.node_types()
            self.edge_types = datamodule.edge_counts()
            self.valency_distribution = datamodule.valency_count(self.max_n_nodes)


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert (
            dataset_infos is not None
        ), "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = "train_smiles_no_h.npy" if remove_h else "train_smiles_h.npy"
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_CHnmr_smiles(atom_decoder, train_dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))
    if evaluate_dataset:
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E
            for k in range(X.shape[0]):
                n = int(paddle.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])
        print(
            "Evaluating the dataset -- number of molecules to evaluate",
            len(all_molecules),
        )
        metrics = compute_molecular_metrics(
            molecule_list=all_molecules,
            train_smiles=train_smiles,
            dataset_info=dataset_infos,
        )
        print(metrics[0])
    return train_smiles


def compute_CHnmr_smiles(atom_decoder, train_dataloader, remove_h):
    print(f"\tConverting CHnmr dataset to SMILES for remove_h={remove_h}...")
    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E
        n_nodes = [int(paddle.sum((X != -1)[j, :])) for j in range(X.shape[0])]
        molecule_list = []
        for k in range(X.shape[0]):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(
                molecule[0], molecule[1], atom_decoder
            )
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1
        if i % 1000 == 0:
            print(
                "\tConverting CHnmr dataset to SMILES {0:.2%}".format(
                    float(i) / len_train
                )
            )
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles
