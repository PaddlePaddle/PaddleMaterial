import os
from collections import defaultdict

# import random
from typing import Callable
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
import pandas as pd
import pgl
from paddle.io import Dataset
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm

from ppmat.datasets.ext_rdkit import build_molecule_with_partial_charges
from ppmat.datasets.ext_rdkit import compute_molecular_metrics
from ppmat.datasets.ext_rdkit import mol2smiles
from ppmat.datasets.utils import numericalize_text
from ppmat.models.digress.distributions import DistributionNodes
from ppmat.models.digress.utils import digressutils as utils

# import pathlib


class RemoveYTransform:
    def __call__(self, data):
        data.y = paddle.zeros((1, 0), dtype=paddle.float32)
        return data


class SelectMuTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data


class SelectHOMOTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data


class CHnmrData:
    """
    data process for Spectrum Graph Molecules
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        vocab_path: str,
        remove_h=False,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
        split_ratio: Tuple = (0.9, 0.05, 0.05),
        stage: Literal["train", "val", "test"] = None,
    ):
        self.path = path
        self.vocab_path = vocab_path
        self.remove_h = remove_h
        self.pre_transform = pre_transform
        self.pre_filter = pre_transform
        self.split_ratio = split_ratio

        self.vocabDim = 256
        self.vocab_to_id = {"<blank>": 0, "<unk>": 1}

        self.get_vocab_id_dict()
        if stage is not None:
            dataset = self.load_data(path, False)
            setattr(self, stage, dataset)
        else:
            dataset = self.load_data(path, True)
            setattr(self, "train", dataset[0])
            setattr(self, "val", dataset[1])
            setattr(self, "test", dataset[2])

    def get_vocab_id_dict(self):
        with open(self.vocab_path, "r", encoding="utf-8") as vocab_file:
            current_id = 2
            for line in vocab_file:
                word, _ = line.strip().split("\t")
                self.vocab_to_id[word] = current_id
                current_id += 1

    def load_data(self, path, split=False):
        dataset = pd.read_csv(path)
        if split:
            datasets = self.split_dataset(dataset)
            dataset_proc = [self.process(target_df) for target_df in datasets]
            return dataset_proc
        return self.process(dataset)

    def split_dataset(self, dataset):
        n_samples = len(dataset)
        n_train = int(self.split_ratio[0] * n_samples)
        n_test = int(self.split_ratio[2] * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(
            dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train]
        )

        data_dir = os.path.dirname(self.path)
        train_file_path = os.path.join(data_dir, "train_paddle.csv")
        val_file_path = os.path.join(data_dir, "val_paddle.csv")
        test_file_path = os.path.join(data_dir, "test_paddle.csv")
        if not os.path.exists(train_file_path):
            train.to_csv(train_file_path)
        if not os.path.exists(val_file_path):
            val.to_csv(val_file_path)
        if not os.path.exists(test_file_path):
            test.to_csv(test_file_path)

        return train, val, test

    def process(self, target_df):
        RDLogger.DisableLog("rdApp.*")

        types = {
            "H": 0,
            "C": 1,
            "N": 2,
            "O": 3,
            "F": 4,
            "P": 5,
            "S": 6,
            "Cl": 7,
            "Br": 8,
            "I": 9,
        }
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        data_list = []
        # for idx, row in tqdm(target_df.iterrows(), total=target_df.shape[0]):
        # for idx in tqdm(range(target_df.shape[0] // 10000)):
        for idx in tqdm(range(2)):
            row = target_df.iloc[idx]
            smiles = row["smiles"]
            tokenized_input = row["tokenized_input"]
            atom_count = row["atom_count"]

            # 将 SMILES 转化为 mol 对象
            mol = Chem.MolFromSmiles(smiles)
            # if mol is None:
            #     continue

            N = mol.GetNumAtoms()

            # 获取原子类型索引
            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            # 获取边信息
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            edge_index = paddle.to_tensor([row, col], dtype=paddle.int64)
            edge_type = paddle.to_tensor(edge_type, dtype=paddle.int64)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1)

            perm = paddle.argsort(edge_index[0] * N + edge_index[1])
            edge_index = paddle.gather(edge_index, perm, axis=1)
            edge_attr = paddle.gather(edge_attr, perm, axis=0)

            x = F.one_hot(paddle.to_tensor(type_idx), num_classes=len(types))
            y = paddle.zeros([1, 0], dtype=paddle.float32)

            if self.remove_h:
                type_idx = paddle.to_tensor(type_idx, dtype=paddle.int64)
                to_keep = type_idx > 0
                edge_index, edge_attr = utils.subgraph(
                    to_keep,
                    edge_index,
                    edge_attr,
                    relabel_nodes=True,
                    num_nodes=len(to_keep),
                )
                x = x[to_keep]
                # Shift onehot encoding to match atom decoder
                x = x[:, 1:]

            # 创建 Data 对象，并包含额外的信息（tokenized_input 和 atom_count）
            data = pgl.Graph(
                num_nodes=x.shape[0],
                edges=edge_index.T.numpy(),
                node_feat={"feat": x.numpy()},
                edge_feat={"feat": edge_attr.numpy()},
            )

            conditionVec = paddle.to_tensor(
                numericalize_text(
                    text=tokenized_input,
                    vocab_to_id=self.vocab_to_id,
                    dim=self.vocabDim,
                ),
                dtype=paddle.int64,
            )
            atom_count = paddle.to_tensor(atom_count, dtype=paddle.int64)

            other_data = {
                "y": y,
                "idx": paddle.to_tensor(idx, dtype=paddle.int64),
                "conditionVec": conditionVec,
                "atom_count": atom_count,
            }

            # 过滤和转换
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append((data, other_data))
        return data_list


class CHnmrDataset(Dataset):
    def __init__(
        self,
        path: Union[str, List[str]],
        vocab_path: str,
        remove_h=False,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
        split_ratio: Tuple = (0.9, 0.05, 0.05),
        stage: Literal["train", "val", "test"] = "test",
        **kwargs,
    ):
        data = CHnmrData(
            path,
            vocab_path,
            remove_h,
            pre_transform,
            pre_filter,
            split_ratio,
            stage,
        )
        self.dataset = getattr(data, stage)
        self.fake_value = paddle.to_tensor(
            1.0
        )  # enable the dataloader to read the graph

        # # TODO: apply transform
        # target = kwargs.get("guidance_target", None)
        # regressor = kwargs.get("regressor", None)
        # if regressor and target == "mu":
        #     transform = SelectMuTransform()
        # elif regressor and target == "homo":
        #     transform = SelectHOMOTransform()
        # elif regressor and target == "both":
        #     transform = None
        # else:
        #     transform = RemoveYTransform()

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # TODO: shuffle
        # random.shuffle(self.dataset)
        for item in self.dataset:
            yield item

    def collate_fn(self, batch):
        graphs, other_datas = map(list, zip(*batch))
        batch_graph = pgl.Graph.batch(graphs)
        batch_graph.tensor()

        new_other_data = defaultdict(list)
        for other_data in other_datas:
            for key, value in other_data.items():
                new_other_data[key].append(value)
        for key, values in new_other_data.items():
            if isinstance(values[0], paddle.Tensor):
                stacked_tensor = paddle.stack(values)
                if len(stacked_tensor.shape) > 1:
                    new_shape = [
                        stacked_tensor.shape[0] * stacked_tensor.shape[1]
                    ] + list(stacked_tensor.shape[2:])
                    new_other_data[key] = stacked_tensor.reshape(new_shape)
        return batch_graph, new_other_data


class CHnmrinfos:
    def __init__(self, dataloaders, cfg, recompute_statistics=False):
        self.remove_h = cfg["Dataset"]["train"]["dataset"]["remove_h"]
        self.need_to_strip = (
            False  # to indicate whether we need to ignore one output from the model
        )

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

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.valency_distribution = paddle.zeros(3 * self.max_n_nodes - 2)
        if not self.remove_h:
            self.valency_distribution[0:6] = paddle.to_tensor(
                [0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012]
            )
        else:
            self.valency_distribution[0:7] = paddle.to_tensor(
                [
                    0.000000000000000000e00,
                    1.856458932161331177e-01,
                    2.707855999469757080e-01,
                    3.008204102516174316e-01,
                    2.362315803766250610e-01,
                    3.544347826391458511e-03,
                    2.972166286781430244e-03,
                ]
            )
        recompute_statistics = True
        if recompute_statistics:
            self.n_nodes = dataloaders.node_counts()
            self.node_types = dataloaders.node_types()
            self.edge_types = dataloaders.edge_counts()
            self.valency_distribution = dataloaders.valency_count(self.max_n_nodes)

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(
        self, dataloader, extra_features, domain_features, conditionDim=0
    ):
        example_batch, other_data = next(iter(dataloader()))
        ex_dense, node_mask = utils.to_dense(
            example_batch.node_feat["feat"],
            example_batch.edges.T,
            example_batch.edge_feat["feat"],
            example_batch.graph_node_id,
        )
        example_data = {
            "X_t": ex_dense.X,
            "E_t": ex_dense.E,
            "y_t": other_data["y"],
            "node_mask": node_mask,
        }

        self.input_dims = {
            "X": example_batch.node_feat["feat"].shape[1],
            "E": example_batch.edge_feat["feat"].shape[1],
            "y": other_data["y"].shape[1] + 1,
        }  # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims["X"] += ex_extra_feat.X.shape[-1]
        self.input_dims["E"] += ex_extra_feat.E.shape[-1]
        self.input_dims["y"] += ex_extra_feat.y.shape[-1]

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims["X"] += ex_extra_molecular_feat.X.shape[-1]
        self.input_dims["E"] += ex_extra_molecular_feat.E.shape[-1]
        self.input_dims["y"] += ex_extra_molecular_feat.y.shape[-1]

        self.input_dims["y"] += conditionDim

        self.output_dims = {
            "X": example_batch.node_feat["feat"].shape[1],
            "E": example_batch.edge_feat["feat"].shape[1],
            "y": 0,
        }


def get_train_smiles(cfg, dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert (
            dataset_infos is not None
        ), "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = os.path.dirname(cfg["Dataset"]["train"]["dataset"]["path"])
    remove_h = cfg["Dataset"]["train"]["dataset"]["remove_h"]
    atom_decoder = dataset_infos.atom_decoder
    # root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = "train_smiles_no_h.npy" if remove_h else "train_smiles_h.npy"
    smiles_path = os.path.join(datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_CHnmr_smiles(atom_decoder, dataloader, remove_h)
        # np.save(smiles_path, np.array(train_smiles))
    if evaluate_dataset:
        all_molecules = []
        for i, data in enumerate(dataloader):
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.graph_node_id
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


def compute_CHnmr_smiles(atom_decoder, dataloader, remove_h):
    print(f"\tConverting CHnmr dataset to SMILES for remove_h={remove_h}...")
    mols_smiles = []
    len_train = len(dataloader)
    invalid = 0
    disconnected = 0
    for i, (data, _) in enumerate(dataloader()):
        print("compute_CHnmr_smiles i:", i)
        dense_data, node_mask = utils.to_dense(
            data.node_feat["feat"],
            data.edges.T,
            data.edge_feat["feat"],
            data.graph_node_id,
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
        for _, molecule in enumerate(molecule_list):
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


class DataLoaderCollection:
    def __init__(self, train_dataloader, val_dataloader=None, test_dataloader=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def node_counts(self, max_nodes_possible=300):
        all_counts = paddle.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data, other_data in loader:
                unique, counts = paddle.unique(data.graph_node_id, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for data, other_data in self.train_dataloader():
            num_classes = data.node_feat["feat"].shape[1]
            break

        counts = paddle.zeros(num_classes)

        for i, (data, other_data) in enumerate(self.train_dataloader()):
            counts += data.node_feat["feat"].sum(axis=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data, other_data in self.train_dataloader():
            num_classes = data.edge_feat["feat"].shape[1]
            break

        d = paddle.zeros(num_classes, dtype=paddle.float32)

        for i, (data, other_data) in enumerate(self.train_dataloader()):
            unique, counts = paddle.unique(data.graph_node_id, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edges.T.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_feat["feat"].sum(axis=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d

    def valency_count(self, max_n_nodes):
        valencies = paddle.zeros(
            3 * max_n_nodes - 2
        )  # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = paddle.to_tensor([0, 1, 2, 3, 1.5])

        for data, other_data in self.train_dataloader():
            n = data.node_feat["feat"].shape[0]

            for atom in range(n):
                edges = data.edge_feat["feat"][data.edges.T[0] == atom]
                edges_total = edges.sum(axis=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.astype("int64").item()] += 1
        valencies = valencies / valencies.sum()
        return valencies
