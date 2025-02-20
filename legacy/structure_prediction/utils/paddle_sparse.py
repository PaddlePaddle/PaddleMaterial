import warnings
from textwrap import indent
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import scipy.sparse
from utils import paddle_aux  # noqa: F401

layouts = ["coo", "csr", "csc"]


def get_layout(layout: Optional[str] = None) -> str:
    if layout is None:
        layout = "coo"
        warnings.warn(
            '`layout` argument unset, using default layout "coo". '
            "This may lead to unexpected behaviour."
        )
    assert layout == "coo" or layout == "csr" or layout == "csc"
    return layout


class SparseStorage(object):
    _row: Optional[paddle.Tensor]
    _rowptr: Optional[paddle.Tensor]
    _col: paddle.Tensor
    _value: Optional[paddle.Tensor]
    _sparse_sizes: Tuple[int, int]
    _rowcount: Optional[paddle.Tensor]
    _colptr: Optional[paddle.Tensor]
    _colcount: Optional[paddle.Tensor]
    _csr2csc: Optional[paddle.Tensor]
    _csc2csr: Optional[paddle.Tensor]

    def __init__(
        self,
        row: Optional[paddle.Tensor] = None,
        rowptr: Optional[paddle.Tensor] = None,
        col: Optional[paddle.Tensor] = None,
        value: Optional[paddle.Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        rowcount: Optional[paddle.Tensor] = None,
        colptr: Optional[paddle.Tensor] = None,
        colcount: Optional[paddle.Tensor] = None,
        csr2csc: Optional[paddle.Tensor] = None,
        csc2csr: Optional[paddle.Tensor] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):
        assert row is not None or rowptr is not None
        assert col is not None
        # assert col.dtype == 'int64'
        assert col.dim() == 1
        col = col
        M: int = 0
        if sparse_sizes is None or sparse_sizes[0] is None:
            if rowptr is not None:
                M = rowptr.size - 1
            elif row is not None and row.size > 0:
                M = int(row.max()) + 1
        else:
            _M = sparse_sizes[0]
            assert _M is not None
            M = _M
            if rowptr is not None:
                assert rowptr.size - 1 == M
            elif row is not None and row.size > 0:
                assert trust_data or int(row.max()) < M
        N: int = 0
        if sparse_sizes is None or sparse_sizes[1] is None:
            if col.size > 0:
                N = int(col.max()) + 1
        else:
            _N = sparse_sizes[1]
            assert _N is not None
            N = _N
            if col.size > 0:
                assert trust_data or int(col.max()) < N
        sparse_sizes = M, N
        # if row is not None:
        #     assert row.dtype == 'int64'
        #     assert row.place == col.place
        #     assert row.dim() == 1
        #     assert row.size == col.size
        #     row = row
        # if rowptr is not None:
        #     assert rowptr.dtype == 'int64'
        #     assert rowptr.place == col.place
        #     assert rowptr.dim() == 1
        #     assert rowptr.size - 1 == sparse_sizes[0]
        #     rowptr = rowptr
        # if value is not None:
        #     assert value.place == col.place
        #     assert value.shape[0] == col.shape[0]
        #     value = value
        # if rowcount is not None:
        #     assert rowcount.dtype == 'int64'
        #     assert rowcount.place == col.place
        #     assert rowcount.dim() == 1
        #     assert rowcount.size == sparse_sizes[0]
        #     rowcount = rowcount
        # if colptr is not None:
        #     assert colptr.dtype == 'int64'
        #     assert colptr.place == col.place
        #     assert colptr.dim() == 1
        #     assert colptr.size - 1 == sparse_sizes[1]
        #     colptr = colptr
        # if colcount is not None:
        #     assert colcount.dtype == 'int64'
        #     assert colcount.place == col.place
        #     assert colcount.dim() == 1
        #     assert colcount.size == sparse_sizes[1]
        #     colcount = colcount
        # if csr2csc is not None:
        #     assert csr2csc.dtype == 'int64'
        #     assert csr2csc.place == col.place
        #     assert csr2csc.dim() == 1
        #     assert csr2csc.size == col.shape[0]
        #     csr2csc = csr2csc
        # if csc2csr is not None:
        #     assert csc2csr.dtype == 'int64'
        #     assert csc2csr.place == col.place
        #     assert csc2csr.dim() == 1
        #     assert csc2csr.size == col.shape[0]
        #     csc2csr = csc2csr
        self._row = row
        self._rowptr = rowptr
        self._col = col
        self._value = value
        self._sparse_sizes = tuple(sparse_sizes)
        self._rowcount = rowcount
        self._colptr = colptr
        self._colcount = colcount
        self._csr2csc = csr2csc
        self._csc2csr = csc2csr
        if not is_sorted and self._col.size > 0:
            idx = paddle.zeros(shape=self._col.size + 1, dtype=self._col.dtype)
            idx[1:] = self.row()
            idx[1:] *= self._sparse_sizes[1]
            idx[1:] += self._col
            if (idx[1:] < idx[:-1]).astype("bool").any():
                # max_value = self._sparse_sizes[0] * self._sparse_sizes[1]
                perm = idx[1:].argsort()
                self._row = self.row()[perm]
                self._col = self._col[perm]
                if value is not None:
                    self._value = value[perm]
                self._csr2csc = None
                self._csc2csr = None

    @classmethod
    def empty(self):
        row = paddle.to_tensor(data=[], dtype="int64")
        col = paddle.to_tensor(data=[], dtype="int64")
        return SparseStorage(
            row=row,
            rowptr=None,
            col=col,
            value=None,
            sparse_sizes=(0, 0),
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=True,
            trust_data=True,
        )

    def has_row(self) -> bool:
        return self._row is not None

    def row(self):
        row = self._row
        if row is not None:
            return row

    #         rowptr = self._rowptr
    #         if rowptr is not None:
    # >>>>>>            row = torch.ops.torch_sparse.ptr2ind(rowptr, self._col.size)
    #             self._row = row
    #             return row
    #         raise ValueError

    def has_rowptr(self) -> bool:
        return self._rowptr is not None

    def rowptr(self) -> paddle.Tensor:
        rowptr = self._rowptr
        if rowptr is not None:
            return rowptr

    #         row = self._row
    #         if row is not None:
    #            rowptr = torch.ops.torch_sparse.ind2ptr(row, self._sparse_sizes[0])
    #             self._rowptr = rowptr
    #             return rowptr
    #         raise ValueError

    def col(self) -> paddle.Tensor:
        return self._col

    def has_value(self) -> bool:
        return self._value is not None

    def value(self) -> Optional[paddle.Tensor]:
        return self._value

    def set_value_(self, value: Optional[paddle.Tensor], layout: Optional[str] = None):
        if value is not None:
            if get_layout(layout) == "csc":
                value = value[self.csc2csr()]
            value = value
            assert value.place == self._col.place
            assert value.shape[0] == self._col.size
        self._value = value
        return self

    def set_value(self, value: Optional[paddle.Tensor], layout: Optional[str] = None):
        if value is not None:
            if get_layout(layout) == "csc":
                value = value[self.csc2csr()]
            value = value
            assert value.place == self._col.place
            assert value.shape[0] == self._col.size
        return SparseStorage(
            row=self._row,
            rowptr=self._rowptr,
            col=self._col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=self._rowcount,
            colptr=self._colptr,
            colcount=self._colcount,
            csr2csc=self._csr2csc,
            csc2csr=self._csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def sparse_sizes(self) -> Tuple[int, int]:
        return self._sparse_sizes

    def sparse_size(self, dim: int) -> int:
        return self._sparse_sizes[dim]

    def sparse_resize(self, sparse_sizes: Tuple[int, int]):
        assert len(sparse_sizes) == 2
        old_sparse_sizes, nnz = self._sparse_sizes, self._col.size
        diff_0 = sparse_sizes[0] - old_sparse_sizes[0]
        rowcount, rowptr = self._rowcount, self._rowptr
        if diff_0 > 0:
            if rowptr is not None:
                rowptr = paddle.concat(
                    x=[
                        rowptr,
                        paddle.full(
                            shape=(diff_0,), fill_value=nnz, dtype=rowptr.dtype
                        ),
                    ]
                )
            if rowcount is not None:
                rowcount = paddle.concat(
                    x=[rowcount, paddle.zeros(shape=diff_0, dtype=rowcount.dtype)]
                )
        elif diff_0 < 0:
            if rowptr is not None:
                rowptr = rowptr[:diff_0]
            if rowcount is not None:
                rowcount = rowcount[:diff_0]
        diff_1 = sparse_sizes[1] - old_sparse_sizes[1]
        colcount, colptr = self._colcount, self._colptr
        if diff_1 > 0:
            if colptr is not None:
                colptr = paddle.concat(
                    x=[
                        colptr,
                        paddle.full(
                            shape=(diff_1,), fill_value=nnz, dtype=colptr.dtype
                        ),
                    ]
                )
            if colcount is not None:
                colcount = paddle.concat(
                    x=[colcount, paddle.zeros(shape=diff_1, dtype=colcount.dtype)]
                )
        elif diff_1 < 0:
            if colptr is not None:
                colptr = colptr[:diff_1]
            if colcount is not None:
                colcount = colcount[:diff_1]
        return SparseStorage(
            row=self._row,
            rowptr=rowptr,
            col=self._col,
            value=self._value,
            sparse_sizes=sparse_sizes,
            rowcount=rowcount,
            colptr=colptr,
            colcount=colcount,
            csr2csc=self._csr2csc,
            csc2csr=self._csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def sparse_reshape(self, num_rows: int, num_cols: int):
        assert num_rows > 0 or num_rows == -1
        assert num_cols > 0 or num_cols == -1
        assert num_rows > 0 or num_cols > 0
        total = self.sparse_size(0) * self.sparse_size(1)
        if num_rows == -1:
            num_rows = total // num_cols
        if num_cols == -1:
            num_cols = total // num_rows
        assert num_rows * num_cols == total
        idx = self.sparse_size(1) * self.row() + self.col()
        row = paddle.floor(paddle.divide(x=idx, y=paddle.to_tensor(num_cols)))
        col = idx % num_cols
        assert row.dtype == "int64" and col.dtype == "int64"
        return SparseStorage(
            row=row,
            rowptr=None,
            col=col,
            value=self._value,
            sparse_sizes=(num_rows, num_cols),
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=True,
            trust_data=True,
        )

    def has_rowcount(self) -> bool:
        return self._rowcount is not None

    def rowcount(self) -> paddle.Tensor:
        rowcount = self._rowcount
        if rowcount is not None:
            return rowcount
        rowptr = self.rowptr()
        rowcount = rowptr[1:] - rowptr[:-1]
        self._rowcount = rowcount
        return rowcount

    def has_colptr(self) -> bool:
        return self._colptr is not None

    def colptr(self) -> paddle.Tensor:
        colptr = self._colptr
        if colptr is not None:
            return colptr

    #         csr2csc = self._csr2csc
    #         if csr2csc is not None:
    # >>>>>>            colptr = torch.ops.torch_sparse.ind2ptr(self._col[csr2csc],
    #                 self._sparse_sizes[1])
    #         else:
    #             colptr = paddle.zeros(shape=self._sparse_sizes[1] + 1, dtype=
    #                 self._col.dtype)
    #             paddle.assign(paddle.cumsum(x=self.colcount(), axis=0), output=
    #                 colptr[1:])
    #         self._colptr = colptr
    #         return colptr

    def has_colcount(self) -> bool:
        return self._colcount is not None

    # def colcount(self) ->paddle.Tensor:
    #     colcount = self._colcount
    #     if colcount is not None:
    #         return colcount
    #     colptr = self._colptr
    #     if colptr is not None:
    #         colcount = colptr[1:] - colptr[:-1]
    #     else:
    #         colcount = scatter_add(paddle.ones_like(x=self._col), self._col,
    #             dim_size=self._sparse_sizes[1])
    #     self._colcount = colcount
    #     return colcount

    def has_csr2csc(self) -> bool:
        return self._csr2csc is not None

    # def csr2csc(self) ->paddle.Tensor:
    #     csr2csc = self._csr2csc
    #     if csr2csc is not None:
    #         return csr2csc
    #     idx = self._sparse_sizes[0] * self._col + self.row()
    #     max_value = self._sparse_sizes[0] * self._sparse_sizes[1]
    #     _, csr2csc = index_sort(idx, max_value)
    #     self._csr2csc = csr2csc
    #     return csr2csc

    def has_csc2csr(self) -> bool:
        return self._csc2csr is not None

    # def csc2csr(self) ->paddle.Tensor:
    #     csc2csr = self._csc2csr
    #     if csc2csr is not None:
    #         return csc2csr
    #     max_value = self._sparse_sizes[0] * self._sparse_sizes[1]
    #     _, csc2csr = index_sort(self.csr2csc(), max_value)
    #     self._csc2csr = csc2csr
    #     return csc2csr

    def is_coalesced(self) -> bool:
        idx = paddle.full(
            shape=(self._col.size + 1,), fill_value=-1, dtype=self._col.dtype
        )
        idx[1:] = self._sparse_sizes[1] * self.row() + self._col
        return bool((idx[1:] > idx[:-1]).astype("bool").all())

    # def coalesce(self, reduce: str='add'):
    #     idx = paddle.full(shape=(self._col.size + 1,), fill_value=-1, dtype
    #         =self._col.dtype)
    #     idx[1:] = self._sparse_sizes[1] * self.row() + self._col
    #     mask = idx[1:] > idx[:-1]
    #     if mask.astype('bool').all():
    #         return self
    #     row = self.row()[mask]
    #     col = self._col[mask]
    #     value = self._value
    #     if value is not None:
    #         ptr = mask.nonzero().flatten()
    #         ptr = paddle.concat(x=[ptr, paddle.full(shape=(1,), fill_value=
    #             value.shape[0], dtype=ptr.dtype)])
    #         value = segment_csr(value, ptr, reduce=reduce)
    #     return SparseStorage(row=row, rowptr=None, col=col, value=value,
    #         sparse_sizes=self._sparse_sizes, rowcount=None, colptr=None,
    #         colcount=None, csr2csc=None, csc2csr=None, is_sorted=True,
    #         trust_data=True)

    def fill_cache_(self):
        self.row()
        self.rowptr()
        self.rowcount()
        self.colptr()
        self.colcount()
        self.csr2csc()
        self.csc2csr()
        return self

    def clear_cache_(self):
        self._rowcount = None
        self._colptr = None
        self._colcount = None
        self._csr2csc = None
        self._csc2csr = None
        return self

    def cached_keys(self) -> List[str]:
        keys: List[str] = []
        if self.has_rowcount():
            keys.append("rowcount")
        if self.has_colptr():
            keys.append("colptr")
        if self.has_colcount():
            keys.append("colcount")
        if self.has_csr2csc():
            keys.append("csr2csc")
        if self.has_csc2csr():
            keys.append("csc2csr")
        return keys

    def num_cached_keys(self) -> int:
        return len(self.cached_keys())

    def copy(self):
        return SparseStorage(
            row=self._row,
            rowptr=self._rowptr,
            col=self._col,
            value=self._value,
            sparse_sizes=self._sparse_sizes,
            rowcount=self._rowcount,
            colptr=self._colptr,
            colcount=self._colcount,
            csr2csc=self._csr2csc,
            csc2csr=self._csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def clone(self):
        row = self._row
        if row is not None:
            row = row.clone()
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.clone()
        col = self._col.clone()
        value = self._value
        if value is not None:
            value = value.clone()
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.clone()
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.clone()
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.clone()
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.clone()
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.clone()
        return SparseStorage(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=rowcount,
            colptr=colptr,
            colcount=colcount,
            csr2csc=csr2csc,
            csc2csr=csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def type(self, dtype: paddle.dtype, non_blocking: bool = False):
        value = self._value
        if value is not None:
            if dtype == value.dtype:
                return self
            else:
                return self.set_value(
                    value.to(dtype=dtype, blocking=not non_blocking), layout="coo"
                )
        else:
            return self

    def type_as(self, tensor: paddle.Tensor, non_blocking: bool = False):
        return self.type(dtype=tensor.dtype, non_blocking=non_blocking)

    def to_device(self, device: str, non_blocking: bool = False):
        if device == self._col.place:
            return self
        row = self._row
        if row is not None:
            row = row.to(device, blocking=not non_blocking)
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.to(device, blocking=not non_blocking)
        col = self._col.to(device, blocking=not non_blocking)
        value = self._value
        if value is not None:
            value = value.to(device, blocking=not non_blocking)
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.to(device, blocking=not non_blocking)
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.to(device, blocking=not non_blocking)
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.to(device, blocking=not non_blocking)
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.to(device, blocking=not non_blocking)
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.to(device, blocking=not non_blocking)
        return SparseStorage(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=rowcount,
            colptr=colptr,
            colcount=colcount,
            csr2csc=csr2csc,
            csc2csr=csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def device_as(self, tensor: paddle.Tensor, non_blocking: bool = False):
        return self.to_device(device=tensor.place, non_blocking=non_blocking)

    def cuda(self):
        new_col = self._col.cuda(blocking=True)
        if new_col.place == self._col.place:
            return self
        row = self._row
        if row is not None:
            row = row.cuda(blocking=True)
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.cuda(blocking=True)
        value = self._value
        if value is not None:
            value = value.cuda(blocking=True)
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.cuda(blocking=True)
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.cuda(blocking=True)
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.cuda(blocking=True)
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.cuda(blocking=True)
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.cuda(blocking=True)
        return SparseStorage(
            row=row,
            rowptr=rowptr,
            col=new_col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=rowcount,
            colptr=colptr,
            colcount=colcount,
            csr2csc=csr2csc,
            csc2csr=csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def pin_memory(self):
        row = self._row
        if row is not None:
            row = row.pin_memory()
        rowptr = self._rowptr
        if rowptr is not None:
            rowptr = rowptr.pin_memory()
        col = self._col.pin_memory()
        value = self._value
        if value is not None:
            value = value.pin_memory()
        rowcount = self._rowcount
        if rowcount is not None:
            rowcount = rowcount.pin_memory()
        colptr = self._colptr
        if colptr is not None:
            colptr = colptr.pin_memory()
        colcount = self._colcount
        if colcount is not None:
            colcount = colcount.pin_memory()
        csr2csc = self._csr2csc
        if csr2csc is not None:
            csr2csc = csr2csc.pin_memory()
        csc2csr = self._csc2csr
        if csc2csr is not None:
            csc2csr = csc2csr.pin_memory()
        return SparseStorage(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=rowcount,
            colptr=colptr,
            colcount=colcount,
            csr2csc=csr2csc,
            csc2csr=csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def is_pinned(self) -> bool:
        is_pinned = True
        row = self._row
        if row is not None:
            is_pinned = is_pinned and "pinned" in str(row.place)
        rowptr = self._rowptr
        if rowptr is not None:
            is_pinned = is_pinned and "pinned" in str(rowptr.place)
        is_pinned = "pinned" in str(self._col.place)
        value = self._value
        if value is not None:
            is_pinned = is_pinned and "pinned" in str(value.place)
        rowcount = self._rowcount
        if rowcount is not None:
            is_pinned = is_pinned and "pinned" in str(rowcount.place)
        colptr = self._colptr
        if colptr is not None:
            is_pinned = is_pinned and "pinned" in str(colptr.place)
        colcount = self._colcount
        if colcount is not None:
            is_pinned = is_pinned and "pinned" in str(colcount.place)
        csr2csc = self._csr2csc
        if csr2csc is not None:
            is_pinned = is_pinned and "pinned" in str(csr2csc.place)
        csc2csr = self._csc2csr
        if csc2csr is not None:
            is_pinned = is_pinned and "pinned" in str(csc2csr.place)
        return is_pinned


# SparseStorage.share_memory_ = share_memory_
# SparseStorage.is_shared = is_shared


# >>>>>>@torch.jit.script
class SparseTensor(object):
    storage: SparseStorage

    def __init__(
        self,
        row: Optional[paddle.Tensor] = None,
        rowptr: Optional[paddle.Tensor] = None,
        col: Optional[paddle.Tensor] = None,
        value: Optional[paddle.Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):
        self.storage = SparseStorage(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=sparse_sizes,
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=is_sorted,
            trust_data=trust_data,
        )

    @classmethod
    def from_storage(self, storage: SparseStorage):
        out = SparseTensor(
            row=storage._row,
            rowptr=storage._rowptr,
            col=storage._col,
            value=storage._value,
            sparse_sizes=storage._sparse_sizes,
            is_sorted=True,
            trust_data=True,
        )
        out.storage._rowcount = storage._rowcount
        out.storage._colptr = storage._colptr
        out.storage._colcount = storage._colcount
        out.storage._csr2csc = storage._csr2csc
        out.storage._csc2csr = storage._csc2csr
        return out

    @classmethod
    def from_edge_index(
        self,
        edge_index: paddle.Tensor,
        edge_attr: Optional[paddle.Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):
        return SparseTensor(
            row=edge_index[0],
            rowptr=None,
            col=edge_index[1],
            value=edge_attr,
            sparse_sizes=sparse_sizes,
            is_sorted=is_sorted,
            trust_data=trust_data,
        )

    @classmethod
    def from_dense(self, mat: paddle.Tensor, has_value: bool = True):
        if mat.dim() > 2:
            index = mat.abs().sum(axis=[i for i in range(2, mat.dim())]).nonzero()
        else:
            index = mat.nonzero()
        index = index.t()
        row = index[0]
        col = index[1]
        value: Optional[paddle.Tensor] = None
        if has_value:
            value = mat[row, col]
        return SparseTensor(
            row=row,
            rowptr=None,
            col=col,
            value=value,
            sparse_sizes=(mat.shape[0], mat.shape[1]),
            is_sorted=True,
            trust_data=True,
        )

    @classmethod
    def from_torch_sparse_coo_tensor(self, mat: paddle.Tensor, has_value: bool = True):
        mat = mat.coalesce()
        index = mat._indices()
        row, col = index[0], index[1]
        value: Optional[paddle.Tensor] = None
        if has_value:
            value = mat.values()
        return SparseTensor(
            row=row,
            rowptr=None,
            col=col,
            value=value,
            sparse_sizes=(mat.shape[0], mat.shape[1]),
            is_sorted=True,
            trust_data=True,
        )

    @classmethod
    def from_torch_sparse_csr_tensor(self, mat: paddle.Tensor, has_value: bool = True):
        rowptr = mat.crow_indices()
        col = mat.col_indices()
        value: Optional[paddle.Tensor] = None
        if has_value:
            value = mat.values()
        return SparseTensor(
            row=None,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=(mat.shape[0], mat.shape[1]),
            is_sorted=True,
            trust_data=True,
        )

    @classmethod
    def eye(
        self,
        M: int,
        N: Optional[int] = None,
        has_value: bool = True,
        dtype: Optional[int] = None,
        device: Optional[str] = None,
        fill_cache: bool = False,
    ):
        N = M if N is None else N
        row = paddle.arange(end=min(M, N))
        col = row
        rowptr = paddle.arange(end=M + 1)
        if M > N:
            rowptr[N + 1 :] = N
        value: Optional[paddle.Tensor] = None
        if has_value:
            value = paddle.ones(shape=row.size, dtype=dtype)
        rowcount: Optional[paddle.Tensor] = None
        colptr: Optional[paddle.Tensor] = None
        colcount: Optional[paddle.Tensor] = None
        csr2csc: Optional[paddle.Tensor] = None
        csc2csr: Optional[paddle.Tensor] = None
        if fill_cache:
            rowcount = paddle.ones(shape=M, dtype="int64")
            if M > N:
                rowcount[N:] = 0
            colptr = paddle.arange(dtype="int64", end=N + 1)
            colcount = paddle.ones(shape=N, dtype="int64")
            if N > M:
                colptr[M + 1 :] = M
                colcount[M:] = 0
            csr2csc = csc2csr = row
        out = SparseTensor(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=(M, N),
            is_sorted=True,
            trust_data=True,
        )
        out.storage._rowcount = rowcount
        out.storage._colptr = colptr
        out.storage._colcount = colcount
        out.storage._csr2csc = csr2csc
        out.storage._csc2csr = csc2csr
        return out

    def copy(self):
        return self.from_storage(self.storage)

    def clone(self):
        return self.from_storage(self.storage.clone())

    def type(self, dtype: paddle.dtype, non_blocking: bool = False):
        value = self.storage.value()
        if value is None or dtype == value.dtype:
            return self
        return self.from_storage(self.storage.astype(dtype))

    def type_as(self, tensor: paddle.Tensor, non_blocking: bool = False):
        return self.type(dtype=tensor.dtype, non_blocking=non_blocking)

    def to_device(self, device: str, non_blocking: bool = False):
        if device == self.device():
            return self
        return self.from_storage(
            self.storage.to_device(device=device, non_blocking=non_blocking)
        )

    def device_as(self, tensor: paddle.Tensor, non_blocking: bool = False):
        return self.to_device(device=tensor.place, non_blocking=non_blocking)

    def coo(self) -> Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
        return self.storage.row(), self.storage.col(), self.storage.value()

    def csr(self) -> Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
        return self.storage.rowptr(), self.storage.col(), self.storage.value()

    def csc(self) -> Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
        perm = self.storage.csr2csc()
        value = self.storage.value()
        if value is not None:
            value = value[perm]
        return self.storage.colptr(), self.storage.row()[perm], value

    def has_value(self) -> bool:
        return self.storage.has_value()

    def set_value_(self, value: Optional[paddle.Tensor], layout: Optional[str] = None):
        self.storage.set_value_(value, layout)
        return self

    def set_value(self, value: Optional[paddle.Tensor], layout: Optional[str] = None):
        return self.from_storage(self.storage.set_value(value, layout))

    def sparse_sizes(self) -> Tuple[int, int]:
        return self.storage.sparse_sizes()

    def sparse_size(self, dim: int) -> int:
        return self.storage.sparse_sizes()[dim]

    def sparse_resize(self, sparse_sizes: Tuple[int, int]):
        return self.from_storage(self.storage.sparse_resize(sparse_sizes))

    def sparse_reshape(self, num_rows: int, num_cols: int):
        return self.from_storage(self.storage.sparse_reshape(num_rows, num_cols))

    def coalesce(self, reduce: str = "sum"):
        return self.from_storage(self.storage.coalesce(reduce))

    def fill_cache_(self):
        self.storage.fill_cache_()
        return self

    def clear_cache_(self):
        self.storage.clear_cache_()
        return self

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self.sizes() != other.sizes():
            return False
        rowptrA, colA, valueA = self.csr()
        rowptrB, colB, valueB = other.csr()
        if valueA is None and valueB is not None:
            return False
        if valueA is not None and valueB is None:
            return False
        if not paddle.equal_all(x=rowptrA, y=rowptrB).item():
            return False
        if not paddle.equal_all(x=colA, y=colB).item():
            return False
        if valueA is None and valueB is None:
            return True
        return paddle.equal_all(x=valueA, y=valueB).item()

    def fill_value_(self, fill_value: float, dtype: Optional[int] = None):
        value = paddle.full(shape=(self.nnz(),), fill_value=fill_value, dtype=dtype)
        return self.set_value_(value, layout="coo")

    def fill_value(self, fill_value: float, dtype: Optional[int] = None):
        value = paddle.full(shape=(self.nnz(),), fill_value=fill_value, dtype=dtype)
        return self.set_value(value, layout="coo")

    def sizes(self) -> List[int]:
        sparse_sizes = self.sparse_sizes()
        value = self.storage.value()
        if value is not None:
            return list(sparse_sizes) + list(tuple(value.shape))[1:]
        else:
            return list(sparse_sizes)

    def size(self, dim: int) -> int:
        return self.sizes()[dim]

    def dim(self) -> int:
        return len(self.sizes())

    def nnz(self) -> int:
        return self.storage.col().size

    def numel(self) -> int:
        value = self.storage.value()
        if value is not None:
            return value.size
        else:
            return self.nnz()

    def density(self) -> float:
        if self.sparse_size(0) == 0 or self.sparse_size(1) == 0:
            return 0.0
        return self.nnz() / (self.sparse_size(0) * self.sparse_size(1))

    def sparsity(self) -> float:
        return 1 - self.density()

    def avg_row_length(self) -> float:
        return self.nnz() / self.sparse_size(0)

    def avg_col_length(self) -> float:
        return self.nnz() / self.sparse_size(1)

    def bandwidth(self) -> int:
        row, col, _ = self.coo()
        return int((row - col).abs_().max())

    def avg_bandwidth(self) -> float:
        row, col, _ = self.coo()
        return float((row - col).abs_().to("float32").mean())

    def bandwidth_proportion(self, bandwidth: int) -> float:
        row, col, _ = self.coo()
        tmp = (row - col).abs_()
        return int((tmp <= bandwidth).sum()) / self.nnz()

    def is_quadratic(self) -> bool:
        return self.sparse_size(0) == self.sparse_size(1)

    def is_symmetric(self) -> bool:
        if not self.is_quadratic():
            return False
        rowptr, col, value1 = self.csr()
        colptr, row, value2 = self.csc()
        if (rowptr != colptr).astype("bool").any() or (col != row).astype("bool").any():
            return False
        if value1 is None or value2 is None:
            return True
        else:
            return bool((value1 == value2).astype("bool").all())

    # def to_symmetric(self, reduce: str='sum'):
    #     N = max(self.size(0), self.size(1))
    #     row, col, value = self.coo()
    #     idx = paddle.full(shape=(2 * col.size + 1,), fill_value=-1, dtype=
    #         col.dtype)
    #     idx[1:row.size + 1] = row
    #     idx[row.size + 1:] = col
    #     idx[1:] *= N
    #     idx[1:row.size + 1] += col
    #     idx[row.size + 1:] += row
    #     idx, perm = paddle.sort(x=idx), paddle.argsort(x=idx)
    #     mask = idx[1:] > idx[:-1]
    #     perm = perm[1:].subtract_(y=paddle.to_tensor(1))
    #     idx = perm[mask]
    #     if value is not None:
    #         ptr = mask.nonzero().flatten()
    #         ptr = paddle.concat(x=[ptr, paddle.full(shape=(1,), fill_value=
    #             perm.shape[0], dtype=ptr.dtype)])
    #         value = paddle.concat(x=[value, value])[perm]
    #         value = segment_csr(value, ptr, reduce=reduce)
    #     new_row = paddle.concat(x=[row, col], axis=0)[idx]
    #     new_col = paddle.concat(x=[col, row], axis=0)[idx]
    #     out = SparseTensor(row=new_row, rowptr=None, col=new_col, value=
    #         value, sparse_sizes=(N, N), is_sorted=True, trust_data=True)
    #     return out

    def detach_(self):
        value = self.storage.value()
        if value is not None:
            value.detach_()
        return self

    def detach(self):
        value = self.storage.value()
        if value is not None:
            value = value.detach()
        return self.set_value(value, layout="coo")

    def requires_grad(self) -> bool:
        value = self.storage.value()
        if value is not None:
            return not value.stop_gradient
        else:
            return False

    def requires_grad_(self, requires_grad: bool = True, dtype: Optional[int] = None):
        if requires_grad and not self.has_value():
            self.fill_value_(1.0, dtype)
        value = self.storage.value()
        if value is not None:
            out_0 = value
            out_0.stop_gradient = not requires_grad
            out_0
        return self

    def pin_memory(self):
        return self.from_storage(self.storage.pin_memory())

    def is_pinned(self) -> bool:
        return "pinned" in str(self.storage.place)

    def device(self):
        return self.storage.col().place

    def cpu(self):
        return self.to_device(
            device=str("cpu").replace("cuda", "gpu"), non_blocking=False
        )

    def cuda(self):
        return self.from_storage(self.storage.cuda(blocking=True))

    def is_cuda(self) -> bool:
        return "gpu" in str(self.storage.col().place)

    def dtype(self):
        value = self.storage.value()
        return value.dtype if value is not None else "float32"

    def is_floating_point(self) -> bool:
        value = self.storage.value()
        return paddle.is_floating_point(x=value) if value is not None else True

    def bfloat16(self):
        return self.type(dtype="bfloat16", non_blocking=False)

    def bool(self):
        return self.type(dtype="bool", non_blocking=False)

    def byte(self):
        return self.type(dtype="uint8", non_blocking=False)

    def char(self):
        return self.type(dtype="int8", non_blocking=False)

    def half(self):
        return self.type(dtype="float16", non_blocking=False)

    def float(self):
        return self.type(dtype="float32", non_blocking=False)

    def double(self):
        return self.type(dtype="float64", non_blocking=False)

    def short(self):
        return self.type(dtype="int16", non_blocking=False)

    def int(self):
        return self.type(dtype="int32", non_blocking=False)

    def long(self):
        return self.type(dtype="int64", non_blocking=False)

    def to_dense(self, dtype: Optional[int] = None) -> paddle.Tensor:
        row, col, value = self.coo()
        if value is not None:
            mat = paddle.zeros(shape=self.sizes(), dtype=value.dtype)
        else:
            mat = paddle.zeros(shape=self.sizes(), dtype=dtype)
        if value is not None:
            mat[row, col] = value
        else:
            mat[row, col] = paddle.ones(shape=self.nnz(), dtype=mat.dtype)
        return mat

    def to_torch_sparse_coo_tensor(self, dtype: Optional[int] = None) -> paddle.Tensor:
        row, col, value = self.coo()
        index = paddle.stack(x=[row, col], axis=0)
        if value is None:
            value = paddle.ones(shape=self.nnz(), dtype=dtype)
        return paddle.sparse.sparse_coo_tensor(
            indices=index, values=value, shape=self.sizes()
        )

    def to_torch_sparse_csr_tensor(self, dtype: Optional[int] = None) -> paddle.Tensor:
        rowptr, col, value = self.csr()
        if value is None:
            value = paddle.ones(shape=self.nnz(), dtype=dtype)
        return paddle.sparse.sparse_csr_tensor(
            crows=rowptr, cols=col, values=value, shape=self.sizes()
        )


def cpu(self) -> SparseTensor:
    return self.device_as(paddle.to_tensor(data=0.0, place="cpu"))


def cuda(self, device: Optional[Union[int, str]] = None, non_blocking: bool = False):
    return self.device_as(paddle.to_tensor(data=0.0, place=device or "cuda"))


def __getitem__(self: SparseTensor, index: Any) -> SparseTensor:
    index = list(index) if isinstance(index, tuple) else [index]
    if (
        len(
            [
                i
                for i in index
                if not isinstance(i, (paddle.Tensor, np.ndarray)) and i == ...
            ]
        )
        > 1
    ):
        raise SyntaxError
    dim = 0
    out = self
    while len(index) > 0:
        item = index.pop(0)
        if isinstance(item, (list, tuple)):
            item = paddle.to_tensor(data=item, place=self.device())
        if isinstance(item, np.ndarray):
            item = paddle.to_tensor(data=item).to(self.device())
        if isinstance(item, int):
            out = paddle.index_select(
                out, index=paddle.to_tensor([item]), axis=dim
            ).squeeze(dim)
            dim += 1
        elif isinstance(item, slice):
            if item.step is not None:
                raise ValueError("Step parameter not yet supported.")
            start = 0 if item.start is None else item.start
            start = self.size(dim) + start if start < 0 else start
            stop = self.size(dim) if item.stop is None else item.stop
            stop = self.size(dim) + stop if stop < 0 else stop
            start_0 = out.shape[dim] + start if start < 0 else start
            out = paddle.slice(out, [dim], [start_0], [start_0 + max(stop - start, 0)])
            dim += 1
        elif paddle.is_tensor(x=item):
            if item.dtype == "bool":
                out = out.masked_select(dim, item)
                dim += 1
            elif item.dtype == "int64":
                out = out.index_select(axis=dim, index=item)
                dim += 1
        elif item == Ellipsis:
            if self.dim() - len(index) < dim:
                raise SyntaxError
            dim = self.dim() - len(index)
        else:
            raise SyntaxError
    return out


def __repr__(self: SparseTensor) -> str:
    i = " " * 6
    row, col, value = self.coo()
    infos = []
    infos += [f"row={indent(row.__repr__(), i)[len(i):]}"]
    infos += [f"col={indent(col.__repr__(), i)[len(i):]}"]
    if value is not None:
        infos += [f"val={indent(value.__repr__(), i)[len(i):]}"]
    infos += [
        f"size={tuple(self.sizes())}, nnz={self.nnz()}, "
        f"density={100 * self.density():.02f}%"
    ]
    infos = ",\n".join(infos)
    i = " " * (len(self.__class__.__name__) + 1)
    return f"{self.__class__.__name__}({indent(infos, i)[len(i):]})"


# SparseTensor.share_memory_ = share_memory_
# SparseTensor.is_shared = is_shared
# SparseTensor.to = to
# SparseTensor.cpu = cpu
# SparseTensor.cuda = cuda
SparseTensor.__getitem__ = __getitem__
SparseTensor.__repr__ = __repr__
ScipySparseMatrix = Union[
    scipy.sparse.coo_matrix, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix
]


# >>>>>>@torch.jit.ignore
def from_scipy(mat: ScipySparseMatrix, has_value: bool = True) -> SparseTensor:
    colptr = None
    if isinstance(mat, scipy.sparse.csc_matrix):
        colptr = paddle.to_tensor(data=mat.indptr).to("int64")
    mat = mat.tocsr()
    rowptr = paddle.to_tensor(data=mat.indptr).to("int64")
    mat = mat.tocoo()
    row = paddle.to_tensor(data=mat.row).to("int64")
    col = paddle.to_tensor(data=mat.col).to("int64")
    value = None
    if has_value:
        value = paddle.to_tensor(data=mat.data)
    sparse_sizes = tuple(mat.shape)[:2]
    storage = SparseStorage(
        row=row,
        rowptr=rowptr,
        col=col,
        value=value,
        sparse_sizes=sparse_sizes,
        rowcount=None,
        colptr=colptr,
        colcount=None,
        csr2csc=None,
        csc2csr=None,
        is_sorted=True,
    )
    return SparseTensor.from_storage(storage)


# >>>>>>@torch.jit.ignore
def to_scipy(
    self: SparseTensor,
    layout: Optional[str] = None,
    dtype: Optional[paddle.dtype] = None,
) -> ScipySparseMatrix:
    assert self.dim() == 2
    layout = get_layout(layout)
    if not self.has_value():
        ones = paddle.ones(shape=self.nnz(), dtype=dtype).numpy()
    if layout == "coo":
        row, col, value = self.coo()
        row = row.detach().cpu().numpy()
        col = col.detach().cpu().numpy()
        value = value.detach().cpu().numpy() if self.has_value() else ones
        return scipy.sparse.coo_matrix((value, (row, col)), self.sizes())
    elif layout == "csr":
        rowptr, col, value = self.csr()
        rowptr = rowptr.detach().cpu().numpy()
        col = col.detach().cpu().numpy()
        value = value.detach().cpu().numpy() if self.has_value() else ones
        return scipy.sparse.csr_matrix((value, col, rowptr), self.sizes())
    elif layout == "csc":
        colptr, row, value = self.csc()
        colptr = colptr.detach().cpu().numpy()
        row = row.detach().cpu().numpy()
        value = value.detach().cpu().numpy() if self.has_value() else ones
        return scipy.sparse.csc_matrix((value, row, colptr), self.sizes())


SparseTensor.from_scipy = from_scipy
SparseTensor.to_scipy = to_scipy
