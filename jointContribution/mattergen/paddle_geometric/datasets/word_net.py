from itertools import chain
from typing import Callable, List, Optional

import paddle
from paddle import Tensor
from paddle_geometric.data import Data, InMemoryDataset, download_url
from paddle_geometric.utils import index_sort


class WordNet18(InMemoryDataset):
    r"""The WordNet18 dataset from the `"Translating Embeddings for Modeling
    Multi-Relational Data"
    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling
    -multi-relational-data>`_ paper,
    containing 40,943 entities, 18 relations and 151,442 fact triplets,
    *e.g.*, furniture includes bed.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = ('https://raw.githubusercontent.com/villmow/'
           'datasets_knowledge_embedding/master/WN18/original')

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.txt', 'valid.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process(self) -> None:
        srcs, dsts, edge_types = [], [], []
        for path in self.raw_paths:
            with open(path) as f:
                edges = [int(x) for x in f.read().split()[1:]]
                edge = paddle.to_tensor(edges, dtype='int64')
                srcs.append(edge[::3])
                dsts.append(edge[1::3])
                edge_types.append(edge[2::3])

        src = paddle.concat(srcs, axis=0)
        dst = paddle.concat(dsts, axis=0)
        edge_type = paddle.concat(edge_types, axis=0)

        train_mask = paddle.zeros(src.shape[0], dtype='bool')
        train_mask[:srcs[0].shape[0]] = True
        val_mask = paddle.zeros(src.shape[0], dtype='bool')
        val_mask[srcs[0].shape[0]:srcs[0].shape[0] + srcs[1].shape[0]] = True
        test_mask = paddle.zeros(src.shape[0], dtype='bool')
        test_mask[srcs[0].shape[0] + srcs[1].shape[0]:] = True

        num_nodes = max(int(src.max()), int(dst.max())) + 1
        _, perm = index_sort(num_nodes * src + dst)

        edge_index = paddle.stack([src[perm], dst[perm]], axis=0)
        edge_type = edge_type[perm]
        train_mask = train_mask[perm]
        val_mask = val_mask[perm]
        test_mask = test_mask[perm]

        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=num_nodes,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


class WordNet18RR(InMemoryDataset):
    r"""The WordNet18RR dataset from the `"Convolutional 2D Knowledge Graph
    Embeddings" <https://arxiv.org/abs/1707.01476>`_ paper, containing 40,943
    entities, 11 relations and 93,003 fact triplets.
    """

    url = ('https://raw.githubusercontent.com/villmow/'
           'datasets_knowledge_embedding/master/WN18RR/original')

    edge2id = {
        '_also_see': 0,
        '_derivationally_related_form': 1,
        '_has_part': 2,
        '_hypernym': 3,
        '_instance_hypernym': 4,
        '_member_meronym': 5,
        '_member_of_domain_region': 6,
        '_member_of_domain_usage': 7,
        '_similar_to': 8,
        '_synset_domain_topic_of': 9,
        '_verb_group': 10,
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.txt', 'valid.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process(self) -> None:
        node2id, idx = {}, 0

        srcs, dsts, edge_types = [], [], []
        for path in self.raw_paths:
            with open(path) as f:
                edges = f.read().split()

                _src = edges[::3]
                _dst = edges[2::3]
                _edge_type = edges[1::3]

                for i in chain(_src, _dst):
                    if i not in node2id:
                        node2id[i] = idx
                        idx += 1

                srcs.append(paddle.to_tensor([node2id[i] for i in _src]))
                dsts.append(paddle.to_tensor([node2id[i] for i in _dst]))
                edge_types.append(
                    paddle.to_tensor([self.edge2id[i] for i in _edge_type]))

        src = paddle.concat(srcs, axis=0)
        dst = paddle.concat(dsts, axis=0)
        edge_type = paddle.concat(edge_types, axis=0)

        train_mask = paddle.zeros(src.shape[0], dtype='bool')
        train_mask[:srcs[0].shape[0]] = True
        val_mask = paddle.zeros(src.shape[0], dtype='bool')
        val_mask[srcs[0].shape[0]:srcs[0].shape[0] + srcs[1].shape[0]] = True
        test_mask = paddle.zeros(src.shape[0], dtype='bool')
        test_mask[srcs[0].shape[0] + srcs[1].shape[0]:] = True

        num_nodes = max(int(src.max()), int(dst.max())) + 1
        _, perm = index_sort(num_nodes * src + dst)

        edge_index = paddle.stack([src[perm], dst[perm]], axis=0)
        edge_type = edge_type[perm]
        train_mask = train_mask[perm]
        val_mask = val_mask[perm]
        test_mask = test_mask[perm]

        data = Data(edge_index=edge_index, edge_type=edge_type,
                    train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
