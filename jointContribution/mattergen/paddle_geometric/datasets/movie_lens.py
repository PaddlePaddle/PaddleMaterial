import os
import os.path as osp
from typing import Callable, List, Optional

import paddle
import numpy as np
from paddle_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class MovieLens(InMemoryDataset):
    r"""A heterogeneous rating dataset, assembled by GroupLens Research from
    the `MovieLens web site <https://movielens.org>`_, consisting of nodes of
    type :obj:`"movie"` and :obj:`"user"`.
    User ratings for movies are available as ground truth labels for the edges
    between the users and the movies :obj:`("user", "rates", "movie")`.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.HeteroData` object and returns a
            transformed version. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.HeteroData` object and returns a
            transformed version. (default: :obj:`None`)
        model_name (str): Name of model used to transform movie titles to node
            features from `Huggingface SentenceTransformer`.
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        model_name: Optional[str] = 'all-MiniLM-L6-v2',
        force_reload: bool = False,
    ) -> None:
        self.model_name = model_name
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join('ml-latest-small', 'movies.csv'),
            osp.join('ml-latest-small', 'ratings.csv'),
        ]

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.model_name}.pdparams'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self) -> None:
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        data = HeteroData()

        df = pd.read_csv(self.raw_paths[0], index_col='movieId')
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genres = paddle.to_tensor(np.array(df['genres'].str.get_dummies('|').values, dtype='float32'))

        model = SentenceTransformer(self.model_name)
        with paddle.no_grad():
            emb = paddle.to_tensor(
                model.encode(df['title'].values, show_progress_bar=True, convert_to_tensor=True)
            )

        data['movie'].x = paddle.concat([emb, genres], axis=-1)

        df = pd.read_csv(self.raw_paths[1])
        user_mapping = {idx: i for i, idx in enumerate(df['userId'].unique())}
        data['user'].num_nodes = len(user_mapping)

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = paddle.to_tensor([src, dst], dtype='int64')

        rating = paddle.to_tensor(df['rating'].values, dtype='int64')
        time = paddle.to_tensor(df['timestamp'].values, dtype='int64')

        data['user', 'rates', 'movie'].edge_index = edge_index
        data['user', 'rates', 'movie'].edge_label = rating
        data['user', 'rates', 'movie'].time = time

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'MovieLens-{self.model_name}({len(self)})'
