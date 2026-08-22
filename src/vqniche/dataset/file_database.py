"""
Filesystem-backed `Database` for `OnDiskDatasetBlob` — one file per tissue section.

WHY THIS EXISTS
---------------
`OnDiskDatasetBlob` stores each section as a single serialized value. With the SQLite backend
that value is one BLOB, and SQLite caps a single value at its compile-time `MAX_LENGTH`, which is
1e9 bytes on this stack. Measured behaviour:

    900 MB  -> OK
    1.05 GB -> sqlite3.InterfaceError: Error binding parameter 0
    2.1 GB  -> OverflowError: BLOB longer than INT_MAX bytes
    PyG's own SQLiteDatabase.insert with a 1.2 GB tensor -> the same InterfaceError

Real sections exceed that comfortably: one `xhb1002-AT10` section is 360,208 cells x 4,949 genes
= 7.13 GB of dense f32 counts, roughly 7x the cap. So the corpus could not be built at all. This
was never hit in testing because every test section was ~5.7 MB (8-12k cells x 169 genes), three
orders of magnitude under.

WHY FILES RATHER THAN `backend='rocksdb'`
-----------------------------------------
RocksDB would have been a one-line switch (PyG already ships `RocksDatabase`), but:

  - its value size is a uint32, capping values at 4 GB — still BELOW our 7.13 GB section, so it
    does not actually solve the problem;
  - `rocksdict` is not installed, and the shared venv must not be modified;
  - an LSM store write-amplifies during compaction, which is pure waste on a corpus that is
    written once and never updated;
  - it is another connection abstraction, so it likely reproduces the thread-affinity problem.

A plain file per section has no value-size limit beyond the filesystem, and Lustre is at its best
on large sequential reads — which is exactly this access pattern. It also has **no thread
affinity**: `sqlite3` connections may only be used on the thread that created them, which is why
`KSectionBlockLoader._fetch_section` has to hand each prefetch thread its own handle. Reading a
file needs no such workaround. Operationally it also means `ls`/`du`/`rsync` work, a single
section can be rebuilt or deleted, and a build that dies at section 400 keeps the 399 already
written.

CONTRACT
--------
`OnDiskDataset.db` (on_disk_dataset.py:83-92) builds the backend as

    cls(path=self.processed_paths[0], schema=self.schema, **kwargs)

passing `name` only for `SQLiteDatabase` subclasses — hence the `(path, schema)` signature here.
`path` is treated as a DIRECTORY, created on demand, holding one file per row. `__len__` must work
on a brand-new empty store because `OnDiskDataset.db` calls `len(self._db)` immediately after
construction.

Only `insert` and `get` are abstract on the `Database` ABC (database.py:123, :180);
`multi_insert` / `multi_get` already fall back to per-item loops, which is the right behaviour for
multi-GB values anyway.
"""

from __future__ import annotations

import os
import re
from typing import Any, List

import torch
from torch_geometric.data.database import Database, Schema

# section_00000.pt — zero-padded so a directory listing sorts by row index.
_FNAME = "section_{:05d}.pt"
_FNAME_RE = re.compile(r"^section_(\d+)\.pt$")


class FileDatabase(Database):
    """
    One file per row, `torch.save` / `torch.load`, no size cap and no open connection.

    Parameters
    ----------
    path : str
        Directory holding the per-row files. Created if absent.
    schema : Schema
        Kept for interface parity with `SQLiteDatabase`. Unused: `torch.save` round-trips the
        serialized value (a `Data` object under `OnDiskDataset`'s default `schema=object`)
        without needing a column mapping.
    """

    def __init__(self, path: str, schema: Schema = object) -> None:
        super().__init__(schema)
        self.path = str(path)
        os.makedirs(self.path, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Connection lifecycle — nothing to hold open
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """No-op: a file store has no connection, hence no thread affinity."""

    def close(self) -> None:
        """No-op. Present so callers can treat every backend uniformly."""

    # ------------------------------------------------------------------ #
    # Row access
    # ------------------------------------------------------------------ #
    def _row_path(self, index: int) -> str:
        return os.path.join(self.path, _FNAME.format(int(index)))

    def insert(self, index: int, data: Any) -> None:
        """
        Write one row.

        Writes to a temporary name and renames, so an interrupted build leaves either a complete
        row or no row — never a truncated file that would fail to load on the next open.
        """
        final = self._row_path(index)
        tmp = final + ".partial"
        torch.save(data, tmp)
        os.replace(tmp, final)

    def get(self, index: int) -> Any:
        """Read one row. Raises `IndexError` for a missing row, matching sequence semantics."""
        p = self._row_path(index)
        if not os.path.exists(p):
            raise IndexError(
                f"No row {index} in {self.path} (expected {os.path.basename(p)}). "
                f"The store holds {len(self)} row(s)."
            )
        # weights_only=False: rows are pickled PyG `Data` objects, not plain tensors.
        return torch.load(p, map_location="cpu", weights_only=False)

    def __len__(self) -> int:
        """
        Number of rows present. Returns 0 for a new empty directory, which
        `OnDiskDataset.db` relies on immediately after construction.

        Counts matching filenames rather than tracking a running total, so the length stays
        correct across process restarts and partial builds. `.partial` files are ignored.
        """
        try:
            names = os.listdir(self.path)
        except FileNotFoundError:
            return 0
        return sum(1 for n in names if _FNAME_RE.match(n))

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def row_indices(self) -> List[int]:
        """Row indices present on disk, ascending — useful for diagnosing a partial build."""
        try:
            names = os.listdir(self.path)
        except FileNotFoundError:
            return []
        out = []
        for n in names:
            m = _FNAME_RE.match(n)
            if m is not None:
                out.append(int(m.group(1)))
        return sorted(out)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)} rows at {self.path})"
