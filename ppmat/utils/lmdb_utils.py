# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic read-only LMDB utilities.

Provides the common read-only LMDB access patterns shared by datasets:

- ``open_lmdb`` / ``close_lmdb``: open/close an LMDB environment with safe
  read-only defaults;
- ``lmdb_size``: number of records (``env.stat()["entries"]``);
- ``lmdb_keys``: list all keys, optionally filtering metadata / non-numeric
  keys;
- ``lmdb_get``: fetch a single record by key (pickle payload);
- ``iter_lmdb``: iterate over (key, value) pairs with multi-format payload
  decoding (zlib -> pickle -> json -> ast);
- ``LmdbReader``: context-manager wrapper bundling the above.

Only the read path is abstracted; writing/building LMDB files stays with the
code that owns the data format, and format-specific value decoding (e.g. the
``__ndarray__`` convention in omol25) stays in the consuming dataset.
"""

import ast
import json
import os
import pickle
import zlib
from typing import Any
from typing import Callable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import lmdb


def open_lmdb(
    file_path: str,
    *,
    subdir: Optional[bool] = None,
    max_readers: int = 126,
) -> lmdb.Environment:
    """Open an LMDB environment read-only with memory-safe defaults.

    Args:
        file_path: Path to the LMDB file or directory.
        subdir: Whether ``file_path`` is a directory. When None (default),
            it is auto-detected from the path.
        max_readers: Maximum number of concurrent readers.

    Returns:
        A read-only :class:`lmdb.Environment`.
    """
    if subdir is None:
        subdir = os.path.isdir(file_path)
    return lmdb.open(
        file_path,
        subdir=subdir,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=max_readers,
    )


def close_lmdb(env: lmdb.Environment) -> None:
    """Close an LMDB environment, ignoring errors (safe to call twice)."""
    try:
        env.close()
    except Exception:
        pass


def lmdb_size(env: lmdb.Environment) -> int:
    """Return the number of records in the environment."""
    return int(env.stat()["entries"])


def lmdb_keys(
    env: lmdb.Environment,
    *,
    skip_meta: bool = True,
    skip_non_numeric: bool = False,
) -> List[str]:
    """List all keys in the environment.

    Args:
        env: An open LMDB environment.
        skip_meta: Whether to skip metadata keys (starting with "__").
        skip_non_numeric: Whether to skip keys that are not integers (useful
            for index-keyed stores; metadata like "length" is dropped).

    Returns:
        List of keys (decoded to str).
    """
    with env.begin() as txn:
        all_keys = [
            k.decode("ascii") if isinstance(k, bytes) else k
            for k in txn.cursor().iternext(values=False)
        ]
    if skip_meta:
        all_keys = [k for k in all_keys if not k.startswith("__")]
    if skip_non_numeric:
        all_keys = [k for k in all_keys if _is_numeric_key(k)]
    return all_keys


def _is_numeric_key(key: str) -> bool:
    try:
        int(key)
        return True
    except ValueError:
        return False


def lmdb_get(
    env: lmdb.Environment, key: str, *, decoder: Optional[Callable] = None
) -> Any:
    """Fetch the record stored under ``key``.

    Args:
        env: An open LMDB environment.
        key: Record key.
        decoder: Optional payload decoder; defaults to ``decode_payload``.

    Returns:
        The deserialized object.

    Raises:
        KeyError: If ``key`` does not exist in the environment.
    """
    key_bytes = key.encode("ascii") if isinstance(key, str) else key
    with env.begin() as txn:
        value = txn.get(key_bytes)
    if value is None:
        raise KeyError(f"Key {key} not found in LMDB.")
    decoder = decoder or decode_payload
    return decoder(value)


def decode_payload(payload: bytes) -> Any:
    """Decode a raw LMDB payload with progressive formats.

    Tries, in order: zlib decompress -> pickle -> json -> ast literal. Returns
    the first successful decode; raises ``ValueError`` if all fail.
    """
    raw = payload
    try:
        raw = zlib.decompress(payload)
    except Exception:
        pass
    try:
        return pickle.loads(raw)
    except Exception:
        pass
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        pass
    try:
        return ast.literal_eval(raw.decode("utf-8"))
    except Exception:
        raise ValueError("Failed to decode LMDB payload with zlib/pickle/json/ast.")


def iter_lmdb(
    env: lmdb.Environment,
    *,
    key_filter: Optional[Callable[[str], bool]] = None,
    decoder: Optional[Callable] = None,
) -> Iterator[Tuple[str, Any]]:
    """Iterate over all (key, decoded_value) pairs in the environment.

    Args:
        env: An open LMDB environment.
        key_filter: Optional predicate on the decoded key (str); keys failing
            it are skipped.
        decoder: Optional payload decoder; defaults to ``decode_payload``.
    """
    decoder = decoder or decode_payload
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key_str = key.decode("ascii") if isinstance(key, bytes) else key
            if key_filter is not None and not key_filter(key_str):
                continue
            yield key_str, decoder(value)


class LmdbReader:
    """Context-manager wrapper around a read-only LMDB environment.

    Example::

        with LmdbReader("data/train.lmdb") as reader:
            print(reader.size)
            for key, value in reader.items():
                ...
            obj = reader.get("0")
    """

    def __init__(
        self,
        file_path: str,
        *,
        subdir: Optional[bool] = None,
        decoder: Optional[Callable] = None,
    ) -> None:
        self.file_path = file_path
        self.subdir = subdir
        self.decoder = decoder
        self._env: Optional[lmdb.Environment] = None

    def __enter__(self) -> "LmdbReader":
        self._env = open_lmdb(self.file_path, subdir=self.subdir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        close_lmdb(self._env)
        self._env = None

    @property
    def env(self) -> lmdb.Environment:
        if self._env is None:
            raise RuntimeError("LmdbReader is not open; use it as a context manager.")
        return self._env

    @property
    def size(self) -> int:
        return lmdb_size(self.env)

    def keys(self, **kwargs) -> List[str]:
        return lmdb_keys(self.env, **kwargs)

    def get(self, key: str) -> Any:
        return lmdb_get(self.env, key, decoder=self.decoder)

    def items(self, **kwargs) -> Iterator[Tuple[str, Any]]:
        return iter_lmdb(self.env, decoder=self.decoder, **kwargs)
