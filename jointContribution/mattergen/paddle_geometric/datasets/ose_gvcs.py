import json
import os
from collections import defaultdict
from typing import Callable, List, Optional

import paddle
from paddle_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_tar,
)


class OSE_GVCS(InMemoryDataset):
    r"""A dataset describing the `Product ecology
    <https://wiki.opensourceecology.org/wiki/Product_Ecologies>`_ of the Open
    Source Ecology's iconoclastic `Global Village Construction Set
    <https://wiki.opensourceecology.org/wiki/
    Global_Village_Construction_Set>`_.
    GVCS is a modular, DIY, low-cost set of blueprints that enables the
    fabrication of the 50 different industrial machines that it takes to
    build a small, sustainable civilization with modern comforts.

    The dataset contains a heterogenous graphs with 50 :obj:`machine` nodes,
    composing the GVCS, and 290 directed edges, each representing one out of
    three relationships between machines.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    machines = [
        '3D Printer', '3D Scanner', 'Aluminum Extractor', 'Backhoe',
        'Bakery Oven', 'Baler', 'Bioplastic Extruder', 'Bulldozer', 'Car',
        'CEB Press', 'Cement Mixer', 'Chipper Hammermill', 'CNC Circuit Mill',
        'CNC Torch Table', 'Dairy Milker', 'Drill Press',
        'Electric Motor Generator', 'Gasifier Burner', 'Hay Cutter',
        'Hay Rake', 'Hydraulic Motor', 'Induction Furnace', 'Industrial Robot',
        'Ironworker', 'Laser Cutter', 'Metal Roller', 'Microcombine',
        'Microtractor', 'Multimachine', 'Nickel-Iron Battery', 'Pelletizer',
        'Plasma Cutter', 'Power Cube', 'Press Forge', 'Rod and Wire Mill',
        'Rototiller', 'Sawmill', 'Seeder', 'Solar Concentrator', 'Spader',
        'Steam Engine', 'Steam Generator', 'Tractor', 'Trencher', 'Truck',
        'Universal Power Supply', 'Universal Rotor', 'Welder',
        'Well-Drilling Rig', 'Wind Turbine'
    ]
    categories = [
        'habitat', 'agriculture', 'industry', 'energy', 'materials',
        'transportation'
    ]
    relationships = ['from', 'uses', 'enables']

    url = 'https://github.com/Wesxdz/ose_gvcs/raw/master/ose_gvcs.tar.gz'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f"{machine.lower().replace(' ', '_')}.json"
            for machine in self.machines
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        data = HeteroData()

        categories = []
        edges = defaultdict(list)

        for path in self.raw_paths:
            with open(path) as f:
                product = json.load(f)
            categories.append(self.categories.index(product['category']))
            for interaction in product['ecology']:
                rt = interaction['relationship']
                if rt not in self.relationships:
                    continue
                dst = interaction['tool']
                if dst not in self.machines:
                    continue
                src = self.machines.index(product['machine'])
                dst = self.machines.index(dst)
                edges[rt].append((src, dst))

        data['machine'].num_nodes = len(categories)
        data['machine'].category = paddle.to_tensor(categories)

        for rel, edge_indices in edges.items():
            edge_index = paddle.to_tensor(edge_indices).transpose([1, 0])
            data['machine', rel, 'machine'].edge_index = edge_index

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
