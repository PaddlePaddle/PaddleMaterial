import os
import pickle
import random

import paddle
from pymatgen.core.structure import Structure

from interatomic_potentials.dataset import StructureData
from interatomic_potentials.dataset import StructureJsonData
from ppmat.models.chgnet.graph import CrystalGraphConverter
from ppmat.utils import io

Tensor = paddle.Tensor
random.seed(100)


# This runnable script shows an example to convert a Structure json dataset to graphs
# and save them.
# So the graph conversion step can be avoided in the future training.
# This is extremely useful if you plan to do hyper-parameter sweeping i.e. learning rate


def make_graphs(
    data: StructureJsonData | StructureData,
    graph_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    """Make graphs from a StructureJsonData dataset.

    Args:
        data (StructureJsonData): a StructureJsonData
        graph_dir (str): a directory to save the graphs
        train_ratio (float): train ratio
        val_ratio (float): val ratio
    """
    os.makedirs(graph_dir, exist_ok=True)
    random.shuffle(data.keys)
    labels = {}
    failed_graphs = []
    print(f"{len(data.keys)} graphs to make")

    for idx, (mp_id, graph_id) in enumerate(data.keys):
        dic = make_one_graph(mp_id, graph_id, data, graph_dir)
        if dic is not False:  # graph made successfully
            if mp_id not in labels:
                labels[mp_id] = {graph_id: dic}
            else:
                labels[mp_id][graph_id] = dic
        else:
            failed_graphs += [(mp_id, graph_id)]
        if idx % 1000 == 0:
            print(idx)

    io.write_json(os.path.join(graph_dir, "labels.json"), labels)
    io.write_json(os.path.join(graph_dir, "failed_graphs.json"), failed_graphs)
    make_partition(labels, graph_dir, train_ratio, val_ratio)


def make_one_graph(mp_id: str, graph_id: str, data, graph_dir) -> dict | bool:
    """Convert a structure to a CrystalGraph and save it."""
    dct = data.data[mp_id].pop(graph_id)
    struct = Structure.from_dict(dct.pop("structure"))
    try:
        graph = data.graph_converter(struct, graph_id=graph_id, mp_id=mp_id)
        graph_data = graph.to_dict()
        for key, value in graph_data.items():
            if isinstance(value, Tensor):
                graph_data[key] = value.numpy()
        # paddle.save(graph, os.path.join(graph_dir, f"{graph_id}.pt"))
        with open(os.path.join(graph_dir, f"{graph_id}.pkl"), "wb") as f:
            pickle.dump(graph_data, f)
    except Exception as e:
        print(e)
        return False
    else:
        return dct


def make_partition(
    data, graph_dir, train_ratio=0.8, val_ratio=0.1, *, partition_with_frame=False
) -> None:
    """Make a train val test partition."""
    random.seed(42)
    if partition_with_frame is False:
        material_ids = list(data)
        random.shuffle(material_ids)
        train_ids, val_ids, test_ids = [], [], []
        for i, mp_id in enumerate(material_ids):
            if i < train_ratio * len(material_ids):
                train_ids.append(mp_id)
            elif i < (train_ratio + val_ratio) * len(material_ids):
                val_ids.append(mp_id)
            else:
                test_ids.append(mp_id)
        partition = {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}
    else:
        raise NotImplementedError("Partition with frame is not implemented yet.")

    io.write_json(os.path.join(graph_dir, "TrainValTest_partition.json"), partition)
    print("Done")


if __name__ == "__main__":
    data = "data/MPtrj_chgnet_100.json"
    graph_dir = "data/MPtrj_chgnet_100_graph"

    converter = CrystalGraphConverter(atom_graph_cutoff=5, bond_graph_cutoff=3)
    data = StructureJsonData(data, graph_converter=converter)
    make_graphs(data, graph_dir)
