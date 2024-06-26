import sys

import numba
import numpy as np
import paddle
import scipy.sparse as sp

sys.path.append("/home/chenxiaoxu02/workspaces/gemnet_paddle/utils")


class DataContainer:
    """
    Parameters
    ----------
        path: str
            Absolute path of the dataset (in npz-format).
        cutoff: float
            Insert edge between atoms if distance is less than cutoff.
        int_cutoff: float
            Cutoff of edge embeddings involved in quadruplet-based message passing.
        triplets_only: bool
            Flag whether to load quadruplet indices as well.
        transforms: list
            Transforms that should be applied on the whole dataset.
        addID: bool
            Whether to add the molecule id to the output.
    """

    def __init__(
        self,
        path,
        cutoff,
        int_cutoff,
        triplets_only=False,
        transforms=None,
        addID=False,
    ):
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
        self.addID = addID
        self.keys = ["N", "Z", "R", "F", "E"]
        if addID:
            self.keys += ["id"]
        self._load_npz(path, self.keys)
        if transforms is None:
            self.transforms = []
        else:
            assert isinstance(transforms, (list, tuple))
            self.transforms = transforms
        for transform in self.transforms:
            transform(self)
        assert self.R is not None
        assert self.N is not None
        assert self.Z is not None
        assert self.E is not None
        assert self.F is not None
        assert len(self.E) > 0
        assert len(self.F) > 0
        self.E = self.E[:, None]
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        self.dtypes, dtypes2 = self.get_dtypes()
        self.dtypes.update(dtypes2)
        self.targets = ["E", "F"]

    def _load_npz(self, path, keys):
        """Load the keys from the file and set as attributes.

        Parameters
        ----------
        path: str
            Absolute path of the dataset (in npz-format).
        keys: list
            Contains keys in the dataset to load and set as attributes.

        Returns
        -------
        None
        """
        with np.load(path, allow_pickle=True) as data:
            for key in keys:
                if key not in data.keys():
                    if key != "F":
                        raise UserWarning(f"Can not find key {key} in the dataset.")
                else:
                    setattr(self, key, data[key])

    @staticmethod
    def _bmat_fast(mats):
        """Combines multiple adjacency matrices into single sparse block matrix.

        Parameters
        ----------
            mats: list
                Has adjacency matrices as elements.

        Returns
        -------
            adj_matrix: sp.csr_matrix
                Combined adjacency matrix (sparse block matrix)
        """
        assert len(mats) > 0
        new_data = np.concatenate([mat.data for mat in mats])
        ind_offset = np.zeros(1 + len(mats), dtype="int32")
        ind_offset[1:] = np.cumsum([tuple(mat.shape)[0] for mat in mats])
        new_indices = np.concatenate(
            [(mats[i].indices + ind_offset[i]) for i in range(len(mats))]
        )
        indptr_offset = np.zeros(1 + len(mats))
        indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
        new_indptr = np.concatenate(
            [(mats[i].indptr[i >= 1 :] + indptr_offset[i]) for i in range(len(mats))]
        )
        shape = ind_offset[-1], ind_offset[-1]
        if len(new_data) == 0:
            return sp.csr_matrix(shape)
        return sp.csr_matrix((new_data, new_indices, new_indptr), shape=shape)

    def __len__(self):
        return len(self.N)

    def __getitem__(self, idx):
        """
        Parameters
        ----------
            idx: array-like
                Ids of the molecules to get.

        Returns
        -------
            data: dict
                nMolecules = len(idx)
                nAtoms = total sum of atoms in the selected molecules
                Contains following keys and values:

                - id: np.ndarray, shape (nMolecules,)
                    Ids of the molecules in the dataset.
                - N: np.ndarray, shape (nMolecules,)
                    Number of atoms in the molecules.
                - Z: np.ndarray, shape (nAtoms,)
                    Atomic numbers (dt. Ordnungszahl).
                - R: np.ndarray, shape (nAtoms,3)
                    Atom positions in °A.
                - F: np.ndarray, shape (nAtoms,3)
                    Forces at the atoms in eV/°A.
                - E: np.ndarray, shape (nMolecules,1)
                    Energy of the molecule in eV.
                - batch_seg: np.ndarray, shape (nAtoms,)
                    Contains the index of the sample the atom belongs to.
                    E.g. [0,0,0, 1,1,1,1, 2,...] where first molecule has 3 atoms,
                    second molecule has 4 atoms etc.
                - id_c: np.ndarray, shape (nEdges,)
                    Indices of edges' source atom.
                - id_a: np.ndarray, shape (nEdges,)
                    Indices of edges' target atom.
                - id_undir: np.ndarray, shape (nEdges,)
                    Indices where the same index denotes opposite edges, c-> and a->c.
                - id_swap: np.ndarray, shape (nEdges,)
                    Indices to map c->a to a->c.
                - id3_expand_ba: np.ndarray, shape (nTriplets,)
                    Indices to map the edges from c->a to b->a in the triplet-based massage passing.
                - id3_reduce_ca: np.ndarray, shape (nTriplets,)
                    Indices to map the edges from c->a to c->a in the triplet-based massage passing.
                - Kidx3: np.ndarray, shape (nTriplets,)
                    Indices to reshape the neighbor indices b->a into a dense matrix.
                - id4_int_a: np.ndarray, shape (nInterEdges,)
                    Indices of the atom a of the interaction edge.
                - id4_int_b: np.ndarray, shape (nInterEdges,)
                    Indices of the atom b of the interaction edge.
                - id4_reduce_ca: np.ndarray, shape (nQuadruplets,)
                    Indices to map c->a to c->a in quadruplet-message passing.
                - id4_expand_db: np.ndarray, shape (nQuadruplets,)
                    Indices to map c->a to d->b in quadruplet-message passing.
                - id4_reduce_intm_ca: np.ndarray, shape (intmTriplets,)
                    Indices to map c->a to intermediate c->a.
                - id4_expand_intm_db: np.ndarray, shape (intmTriplets,)
                    Indices to map d->b to intermediate d->b.
                - id4_reduce_intm_ab: np.ndarray, shape (intmTriplets,)
                    Indices to map b-a to intermediate b-a of the quadruplet's part c->a-b.
                - id4_expand_intm_ab: np.ndarray, shape (intmTriplets,)
                    Indices to map b-a to intermediate b-a of the quadruplet's part a-b<-d.
                - id4_reduce_cab: np.ndarray, shape (nQuadruplets,)
                    Indices to map from intermediate c->a to quadruplet c->a.
                - id4_expand_abd: np.ndarray, shape (nQuadruplets,)
                    Indices to map from intermediate d->b to quadruplet d->b.
                - Kidx4: np.ndarray, shape (nTriplets,)
                    Indices to reshape the neighbor indices d->b into a dense matrix.
        """
        if isinstance(idx, (int, np.int64, np.int32)):
            idx = [idx]
        if isinstance(idx, tuple):
            idx = list(idx)
        if isinstance(idx, slice):
            idx = np.arange(idx.start, min(idx.stop, len(self)), idx.step)
        data = {}
        if self.addID:
            data["id"] = self.id[idx]
        data["E"] = self.E[idx]
        data["N"] = self.N[idx]
        data["batch_seg"] = np.repeat(np.arange(len(idx), dtype=np.int32), data["N"])
        data["Z"] = np.zeros(np.sum(data["N"]), dtype=np.int32)
        data["R"] = np.zeros([np.sum(data["N"]), 3], dtype=np.float32)
        data["F"] = np.zeros([np.sum(data["N"]), 3], dtype=np.float32)
        nend = 0
        adj_matrices = []
        adj_matrices_int = []
        for k, i in enumerate(idx):
            n = data["N"][k]
            nstart = nend
            nend = nstart + n
            s, e = self.N_cumsum[i], self.N_cumsum[i + 1]
            data["F"][nstart:nend] = self.F[s:e]
            data["Z"][nstart:nend] = self.Z[s:e]
            R = self.R[s:e]
            data["R"][nstart:nend] = R
            D_ij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            adj_mat = sp.csr_matrix(D_ij <= self.cutoff)
            adj_mat -= sp.eye(n, dtype=np.bool_)
            adj_matrices.append(adj_mat)
            if not self.triplets_only:
                adj_mat = sp.csr_matrix(D_ij <= self.int_cutoff)
                adj_mat -= sp.eye(n, dtype=np.bool_)
                adj_matrices_int.append(adj_mat)
        idx_data = {key: None for key in self.index_keys if key != "batch_seg"}
        adj_matrix = self._bmat_fast(adj_matrices)
        idx_t, idx_s = adj_matrix.nonzero()
        if not self.triplets_only:
            adj_matrix_int = self._bmat_fast(adj_matrices_int)
            idx_int_t, idx_int_s = adj_matrix_int.nonzero()
        if len(idx_t) == 0:
            for key in idx_data.keys():
                data[key] = np.array([], dtype="int32")
            return self.convert_to_tensor(data)
        edges = np.stack([idx_t, idx_s], axis=0)
        mask = edges[0] < edges[1]
        edges = edges[:, mask]
        edges = np.concatenate([edges, edges[::-1]], axis=-1).astype("int32")
        idx_t, idx_s = edges[0], edges[1]
        indices = np.arange(len(mask) / 2, dtype="int32")
        idx_data["id_undir"] = np.concatenate(2 * [indices], axis=-1).astype("int32")
        idx_data["id_c"] = idx_s
        idx_data["id_a"] = idx_t
        if not self.triplets_only:
            idx_data["id4_int_a"] = idx_int_t
            idx_data["id4_int_b"] = idx_int_s
        N_undir_edges = int(len(idx_s) / 2)
        ind = np.arange(N_undir_edges, dtype="int32")
        id_swap = np.concatenate([ind + N_undir_edges, ind])
        idx_data["id_swap"] = id_swap
        edge_ids = sp.csr_matrix(
            (np.arange(len(idx_s)), (idx_t, idx_s)),
            shape=tuple(adj_matrix.shape),
            dtype="int32",
        )
        id3_expand_ba, id3_reduce_ca = self.get_triplets(idx_s, idx_t, edge_ids)
        id3_reduce_ca = id_swap[id3_reduce_ca]
        if len(id3_reduce_ca) > 0:
            idx_sorted = np.argsort(id3_reduce_ca)
            id3_reduce_ca = id3_reduce_ca[idx_sorted]
            id3_expand_ba = id3_expand_ba[idx_sorted]
            _, K = np.unique(id3_reduce_ca, return_counts=True)
            idx_data["Kidx3"] = DataContainer.ragged_range(K)
        else:
            idx_data["Kidx3"] = np.array([], dtype="int32")
        idx_data["id3_expand_ba"] = id3_expand_ba
        idx_data["id3_reduce_ca"] = id3_reduce_ca
        if self.triplets_only:
            data.update(idx_data)
            return self.convert_to_tensor(data)
        output = self.get_quadruplets(
            idx_s, idx_t, adj_matrix, edge_ids, idx_int_s, idx_int_t
        )
        (
            id4_reduce_ca,
            id4_expand_db,
            id4_reduce_cab,
            id4_expand_abd,
            id4_reduce_intm_ca,
            id4_expand_intm_db,
            id4_reduce_intm_ab,
            id4_expand_intm_ab,
        ) = output
        if len(id4_reduce_ca) > 0:
            sorted_idx = np.argsort(id4_reduce_ca)
            id4_reduce_ca = id4_reduce_ca[sorted_idx]
            id4_expand_db = id4_expand_db[sorted_idx]
            id4_reduce_cab = id4_reduce_cab[sorted_idx]
            id4_expand_abd = id4_expand_abd[sorted_idx]
            _, K = np.unique(id4_reduce_ca, return_counts=True)
            idx_data["Kidx4"] = DataContainer.ragged_range(K)
        else:
            idx_data["Kidx4"] = np.array([], dtype="int32")
        idx_data["id4_reduce_ca"] = id4_reduce_ca
        idx_data["id4_expand_db"] = id4_expand_db
        idx_data["id4_reduce_cab"] = id4_reduce_cab
        idx_data["id4_expand_abd"] = id4_expand_abd
        idx_data["id4_reduce_intm_ca"] = id4_reduce_intm_ca
        idx_data["id4_expand_intm_db"] = id4_expand_intm_db
        idx_data["id4_reduce_intm_ab"] = id4_reduce_intm_ab
        idx_data["id4_expand_intm_ab"] = id4_expand_intm_ab
        data.update(idx_data)
        return self.convert_to_tensor(data)

    @staticmethod
    def get_triplets(idx_s, idx_t, edge_ids):
        """
        Get triplets c -> a <- b
        """
        id3_expand_ba = edge_ids[idx_s].data.astype("int32").flatten()
        id3_reduce_ca = edge_ids[idx_s].tocoo().row.astype("int32").flatten()
        id3_i = idx_t[id3_reduce_ca]
        id3_k = idx_s[id3_expand_ba]
        mask = id3_i != id3_k
        id3_expand_ba = id3_expand_ba[mask]
        id3_reduce_ca = id3_reduce_ca[mask]
        return id3_expand_ba, id3_reduce_ca

    @staticmethod
    def get_quadruplets(idx_s, idx_t, adj_matrix, edge_ids, idx_int_s, idx_int_t):
        """
        c -> a - b <- d where D_ab <= int_cutoff; D_ca & D_db <= cutoff
        """
        nNeighbors_t = adj_matrix[idx_int_t].sum(axis=1).A1.astype("int32")
        nNeighbors_s = adj_matrix[idx_int_s].sum(axis=1).A1.astype("int32")
        id4_reduce_intm_ca = edge_ids[idx_int_t].data.astype("int32").flatten()
        id4_expand_intm_db = edge_ids[idx_int_s].data.astype("int32").flatten()
        id4_reduce_cab = DataContainer.repeat_blocks(nNeighbors_t, nNeighbors_s)
        id4_reduce_ca = id4_reduce_intm_ca[id4_reduce_cab]
        N = np.repeat(nNeighbors_t, nNeighbors_s)
        id4_expand_abd = np.repeat(np.arange(len(id4_expand_intm_db)), N)
        id4_expand_db = id4_expand_intm_db[id4_expand_abd]
        id4_reduce_intm_ab = np.repeat(np.arange(len(idx_int_t)), nNeighbors_t)
        id4_expand_intm_ab = np.repeat(np.arange(len(idx_int_t)), nNeighbors_s)
        idx_c = idx_s[id4_reduce_ca]
        idx_a = idx_t[id4_reduce_ca]
        idx_b = idx_t[id4_expand_db]
        idx_d = idx_s[id4_expand_db]
        mask1 = idx_c != idx_b
        mask2 = idx_a != idx_d
        mask3 = idx_c != idx_d
        mask = mask1 * mask2 * mask3
        id4_reduce_ca = id4_reduce_ca[mask]
        id4_expand_db = id4_expand_db[mask]
        id4_reduce_cab = id4_reduce_cab[mask]
        id4_expand_abd = id4_expand_abd[mask]
        return (
            id4_reduce_ca,
            id4_expand_db,
            id4_reduce_cab,
            id4_expand_abd,
            id4_reduce_intm_ca,
            id4_expand_intm_db,
            id4_reduce_intm_ab,
            id4_expand_intm_ab,
        )

    def convert_to_tensor(self, data):
        for key in data:
            data[key] = paddle.to_tensor(data=data[key], dtype=self.dtypes[key])
        return data

    def get_dtypes(self):
        """
        Returns
        -------
            dtypes: tuple
                (dtypes_input, dtypes_target) TF input types for the inputs and targets
                stored in dicts.
        """
        dtypes_input = {}
        if self.addID:
            dtypes_input["id"] = "int64"
        dtypes_input["Z"] = "int64"
        dtypes_input["N"] = "int64"
        dtypes_input["R"] = "float32"
        for key in self.index_keys:
            dtypes_input[key] = "int64"
        dtypes_target = {}
        dtypes_target["E"] = "float32"
        dtypes_target["F"] = "float32"
        return dtypes_input, dtypes_target

    @staticmethod
    @numba.njit(nogil=True)
    def repeat_blocks(sizes, repeats):
        """Repeat blocks of indices.
        From https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

        Examples
        --------
            sizes = [1,3,2] ; repeats = [3,2,3]
            Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
            sizes = [0,3,2] ; repeats = [3,2,3]
            Return: [0 1 2 0 1 2  3 4 3 4 3 4]
            sizes = [2,3,2] ; repeats = [2,0,2]
            Return: [0 1 0 1  5 6 5 6]
        """
        a = np.arange(np.sum(sizes))
        indices = np.empty((sizes * repeats).sum(), dtype=np.int32)
        start = 0
        oi = 0
        for i, size in enumerate(sizes):
            end = start + size
            for _ in range(repeats[i]):
                oe = oi + size
                indices[oi:oe] = a[start:end]
                oi = oe
            start = end
        return indices

    @staticmethod
    @numba.njit(nogil=True)
    def ragged_range(sizes):
        """
        -------
        Example
        -------
            sizes = [1,3,2] ;
            Return: [0  0 1 2  0 1]
        """
        a = np.arange(sizes.max())
        indices = np.empty(sizes.sum(), dtype=np.int32)
        start = 0
        for size in sizes:
            end = start + size
            indices[start:end] = a[:size]
            start = end
        return indices
