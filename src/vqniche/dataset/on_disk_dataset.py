"""
On-disk (streaming) DatasetBlob for VQNiche.

Motivation
----------
`InMemoryDatasetBlob` collates every tissue section into a single
`dataset_blob.pt` and pulls the whole thing into RAM at
`self.load(self.processed_paths[0])` (in_memory_dataset_blob.py:130).
For a corpus like `hst_corpus_110m` (~1.1e8 cells) that single object is
~0.6 TB, dominated by:
  - per-graph Laplacian eigenvectors: 6 graphs x N x 128 f32  ~= 315 GB
    (n_neighs_list=[6,8,12,16,20,24], lm_eigvecs=128)
  - the DENSIFIED count matrix: sparse_mx_to_float_tensor
    (type_conversions.py:49) calls `.toarray()`, N x G f32 ~= 205 GB.
No cluster node holds it, and building it produces one unmanageable file.

`OnDiskDatasetBlob` keeps the SAME per-section preprocessing but stores
each tissue section as a row in an on-disk SQLite-backed
`torch_geometric.data.OnDiskDataset`, fetched lazily via `get(idx)` (which
is `deserialize(self.db.get(idx))`). Peak memory becomes
"the largest section (a few GB) + the active NeighborLoader sub-batch"
instead of the entire corpus.

Design decisions
----------------
1. Storage unit = ONE TISSUE SECTION (= one silver .h5ad = one PyG `Data`).
   The spatial neighbor graph is built WITHIN a section
   (in_memory_dataset_blob.py:416, spatial_neighbors); no cross-section
   edges exist, so a section is a self-contained connected component and
   the natural shard. Sharding finer would cut spatial edges.

2. Reuse `InMemoryDatasetBlob.process_anndata_batch()` VERBATIM for the
   per-section Data build. We inherit from it so that method, the
   silver/gold path properties (raw_dir / raw_file_names), and
   `_derive_adata_batch_id` come for free.

3. Use PyG's REAL OnDiskDataset machinery (schema=object) rather than a
   hand-rolled store. Verified against PyG 2.6.1:
   `extend([serialize(d)])` writes rows; `get(idx)` lazily deserializes one
   `Data` (python-list attrs like `cell_id` survive the default pickle
   schema). NOTE: OnDiskDataset.__init__ does NOT accept `pre_transform`
   (signature is root, transform, pre_filter, backend, schema, log).

3b. CONTAINER: the default is `backend="file"` (`file_database.FileDatabase`,
   one file per section), NOT sqlite. SQLite caps a single value at its
   compile-time MAX_LENGTH — 1e9 bytes on this stack, measured: 900 MB OK,
   1.05 GB InterfaceError, 2.1 GB OverflowError — while a real
   `xhb1002-AT10` section is 360,208 x 4,949 = 7.13 GB dense, roughly 7x
   over. So the sqlite container cannot hold the corpus at all; it went
   unnoticed because every test section was ~5.7 MB. `backend="sqlite"`
   remains selectable and existing blobs keep opening (the manifest's
   `container` field defaults to "sqlite" when absent). RocksDB was
   rejected: its value size is a uint32, capping at 4 GB — still below
   7.13 GB — and it is not installed. See file_database.py for the full
   rationale.

4. gold subdir is distinct (`on-disk-PyG-dataset-blob`) so an on-disk blob
   never collides with an in-memory blob of the same dataset.

Consuming sections for training
-------------------------------
`InMemoryDataModule` wants a single concatenated `Data`, which defeats
streaming. Stream instead by iterating sections (`iter_sections`) and
running a NeighborLoader per section, so only one section is resident.

CAVEAT (verified on PyG 2.6.1): `NeighborLoader` raises
"invalid feature tensor type (got 'list')" on python-list node attrs
(`cell_id`, `obs_batch`). This is NOT specific to streaming — the
in-memory path has the same constraint — but the streaming loop must strip
list-typed attrs before sampling. `sampling_view()` returns a shallow copy
of a section carrying only tensor node attributes, safe to feed to
NeighborLoader; the stripped metadata is recoverable by section index.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
import random
from pathlib import Path
from typing import Callable, List, Optional, Iterator, Tuple

import anndata as ad
import pandas as pd
import torch
from torch_geometric.data import Data, OnDiskDataset

from .file_database import FileDatabase
from .in_memory_dataset_blob import InMemoryDatasetBlob


# Node-level attributes that are python lists / non-tensors and therefore
# must be removed from a section before NeighborLoader subgraph sampling.
# (Set at in_memory_dataset_blob.py:522 (cell_id) and :556 (obs_batch).)
_NON_TENSOR_NODE_ATTRS = ("cell_id", "obs_batch")


# Transforms that must make ONE decision for the whole corpus and are
# therefore invalid as a per-section `section_transform`.
#
# `SubsetHVG` picks the top-n highly variable genes from whatever Data it is
# handed (dataset/transforms.py:516-530). Applied per section it selects a
# DIFFERENT gene set per section; the resulting `x` tensors all have width
# n_genes, so concatenation succeeds without complaint even though column j
# means a different gene in different sections. The model has one weight per
# input column shared across all sections, so that column is then trained on
# two unrelated genes at once. Nothing raises.
#
# NOTE this is NOT a streaming-only defect. The in-memory path has the same
# scoping: `initialize_dataset_blob` hands the composed transform to the
# dataset as `transform=`, and PyG applies it in `Dataset.__getitem__`
# (torch_geometric/data/dataset.py:291) — i.e. PER SECTION, before
# `initialize_databatch` collates. So enabling HVG there would scramble
# columns in exactly the same way. It has never fired only because
# `apply_hvg` defaults to False (the gene panels are already curated), not
# because collation protected it.
#
# Fixing it properly means deciding the gene set once, over the whole corpus,
# and applying the SAME indices to every section. Until that exists, refuse
# the transform here rather than silently corrupting features. That makes the
# streaming loader stricter than the in-memory path, deliberately: it is the
# path being built for multi-panel corpora, where HVG is the documented
# recommendation (run_squint.py:868-869).
_GLOBAL_SCOPE_TRANSFORMS = {
    "SubsetHVG": (
        "picks highly variable genes from the Data it is given, so per-section "
        "application selects a different gene set per section and column j "
        "stops meaning the same gene across the corpus"
    ),
}


def _iter_transforms(transform):
    """Yield a transform and, for a Compose, each of its members."""
    if transform is None:
        return
    inner = getattr(transform, "transforms", None)
    if inner is not None:
        for t in inner:
            yield from _iter_transforms(t)
    else:
        yield transform


def _reject_global_scope_transforms(transform) -> None:
    """
    Raise if `transform` contains a transform that must decide globally.

    Converts a silent feature-scrambling bug into an immediate, explanatory
    error. See `_GLOBAL_SCOPE_TRANSFORMS`.
    """
    offenders = [
        (type(t).__name__, _GLOBAL_SCOPE_TRANSFORMS[type(t).__name__])
        for t in _iter_transforms(transform)
        if type(t).__name__ in _GLOBAL_SCOPE_TRANSFORMS
    ]
    if not offenders:
        return
    lines = "\n".join(f"  - {name}: {why}" for name, why in offenders)
    raise ValueError(
        "section_transform contains transform(s) that must make a single "
        "decision for the whole corpus and cannot be applied per section:\n"
        f"{lines}\n"
        "These would not raise at runtime — the shapes stay consistent while "
        "the meaning of each feature column diverges between sections — so "
        "they are rejected here instead.\n"
        "For highly variable genes: either train on the full panel "
        "(apply_hvg=False, the default), or select the gene set once and "
        "apply the same indices to every section. Split transforms "
        "(RandomNodeSplit, SpatialBatchSplit) and SetExperimentDataKeys are "
        "per-section by design and remain fine here."
    )


def _section_batch_label(section: Data) -> str:
    """
    The section's batch label — the value of `adata.uns['batch']`.

    `process_anndata_batch` stores it as `obs_batch`, the per-section value
    broadcast to every cell (in_memory_dataset_blob.py:554-557), so every
    element is identical and the first one is the label. Falls back to the
    integer `adata_batch_id` when `uns['batch']` was absent at build time
    (that field is left unset in exactly that case).

    This is the same identity `initialize_databatch` densifies via
    `build_batch_one_hot_from_obs`, so streaming and in-memory agree on what
    "which batch is this cell from" means.
    """
    obs_batch = section.get("obs_batch", None) if hasattr(section, "get") else None
    if obs_batch is None:
        obs_batch = getattr(section, "obs_batch", None)
    if obs_batch is not None and len(obs_batch) > 0:
        return str(obs_batch[0])
    return str(int(section.adata_batch_id))


class OnDiskDatasetBlob(OnDiskDataset):
    """
    Streaming counterpart of `InMemoryDatasetBlob`. One section per SQLite
    row, fetched lazily by `get(idx)`; storage/fetch use OnDiskDataset's
    SQLite backend.

    COMPOSITION, not multiple inheritance. We inherit ONLY `OnDiskDataset`
    and BORROW the per-section preprocessing methods + silver-side path
    properties from `InMemoryDatasetBlob` as class attributes below. This is
    deliberate: mixing `InMemoryDatasetBlob` into the base list breaks the
    cooperative `__init__` chain — `OnDiskDataset.__init__` calls
    `super().__init__(..., log=...)`, which (whatever the order) routes into
    `InMemoryDatasetBlob.__init__` (no `log` kwarg -> TypeError) or into
    `InMemoryDataset.len` via `append()` (`AttributeError: 'slices'`,
    because that path expects a collated in-memory blob). Borrowing the
    methods sidesteps the MRO entirely: the borrowed functions only touch
    `self.gene_panel`, `self.label_names`, `self.graph_kwargs`, etc., all of
    which we set in `__init__`.
    """

    # Extend PyG's backend registry with a filesystem container.
    # `OnDiskDataset.__init__` validates `backend not in self.BACKENDS` and raises, so
    # adding to the dict on the subclass is the supported way in. `OnDiskDataset.db`
    # then builds it as `cls(path=processed_paths[0], schema=..)`, passing `name` only
    # for SQLiteDatabase subclasses — which is why FileDatabase takes `(path, schema)`.
    #
    # 'file' is the DEFAULT for new blobs: SQLite caps a single value at 1e9 bytes and
    # real sections reach 7.13 GB, so the sqlite container cannot hold the corpus at
    # all. It is kept selectable so existing blobs stay readable.
    BACKENDS = {**OnDiskDataset.BACKENDS, "file": FileDatabase}

    # --- Borrow ALL of InMemoryDatasetBlob's OWN preprocessing methods /
    #     silver-side path properties, EXCEPT the ones we override here.
    #     `process_anndata_batch` calls sibling helpers
    #     (save_adjacency_matrix_to_edgelist, build_*_embeddings, ...), so we
    #     pull every own-attribute of the blob class rather than hand-listing
    #     — new helpers get picked up automatically. We skip `__init__`,
    #     `process`, `processed_dir`, `processed_file_names` (overridden
    #     below) and dunders. `raw_dir`/`raw_file_names`/`raw_paths` ARE
    #     borrowed so the silver glob points at <root>/silver/<name>/.
    _BORROW_SKIP = {"__init__", "process", "processed_dir", "processed_file_names"}
    for _name, _attr in vars(InMemoryDatasetBlob).items():
        if _name.startswith("__"):
            continue
        if _name in _BORROW_SKIP:
            continue
        locals()[_name] = _attr
    del _name, _attr

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        name: str = "sss2-1b_1p",
        feature_names: List[str] = [],
        label_names: List[str] = [],
        graph_kwargs: dict = {},
        data_directory_path: Optional[str | Path] = "/lustre/scratch126/cellgen/team361/DATASETS",
        transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        overwrite: bool = False,
        software_paths: dict = {},
        backend: str = "file",
    ) -> None:
        # Mirror InMemoryDatasetBlob.__init__ attribute setup WITHOUT calling
        # it (its super().__init__ is InMemoryDataset's, which eagerly loads a
        # single collated blob — exactly what we're replacing). Attribute
        # names/derivations copied from in_memory_dataset_blob.py:100-118.
        self.name = name
        self.feature_names = feature_names
        self.label_names = [ln.split("=")[0] for ln in label_names]
        self.label_keys = [ln.split("=")[1] for ln in label_names]
        self.graph_kwargs = graph_kwargs
        if isinstance(data_directory_path, str):
            data_directory_path = Path(data_directory_path)
        self.data_directory_path = data_directory_path
        self.software_paths = software_paths

        # force_reload is consulted by OnDiskDataset._process via the base
        # Dataset; store it so `process()` gating matches PyG semantics.
        self.force_reload = overwrite

        # Drive PyG's OnDiskDataset (SQLite), NOT InMemoryDataset. Its
        # __init__ opens/creates the DB and calls _process() -> process()
        # when the DB is absent or force_reload is set. schema=object =>
        # rows are pickled Data objects (list attrs survive).
        OnDiskDataset.__init__(
            self,
            root=self.processed_dir,
            transform=transform,
            pre_filter=pre_filter,
            backend=backend,
            schema=object,
            log=True,
        )

        # Restore shared metadata sidecars written by process().
        self._load_sidecars()

    # ------------------------------------------------------------------ #
    # Paths (distinct gold subdir; raw_dir / raw_file_names inherited)
    # ------------------------------------------------------------------ #
    @property
    def processed_dir(self) -> str:
        gold = self.data_directory_path / "gold"
        return str(gold / "on-disk-PyG-dataset-blob" / self.name)

    @property
    def processed_file_names(self) -> List[str]:
        """
        Artifacts whose presence makes `files_exist()` skip `process()`.

        Backend-dependent, because the container decides what the store is called:
        SQLite writes a single `sqlite.db` file, while `FileDatabase` writes a
        `sections/` DIRECTORY (which is `processed_paths[0]`, i.e. the `path` handed to
        the backend by `OnDiskDataset.db`).

        `manifest.json` stays in the list for both. It is written only at the very END
        of `process()`, so an interrupted build leaves the manifest absent and the next
        instantiation rebuilds rather than opening a half-populated store.
        """
        store = "sections" if self.backend == "file" else f"{self.backend}.db"
        return [store, "manifest.json"]

    @property
    def _meta_dir(self) -> Path:
        return Path(self.processed_dir)

    # ------------------------------------------------------------------ #
    # Build: SQLite row per section (streamed, never concatenated)
    # ------------------------------------------------------------------ #
    def process(self) -> None:
        """
        Build the on-disk blob in TWO passes, mirroring
        `InMemoryDatasetBlob.process` (in_memory_dataset_blob.py:199-349) but
        writing one SQLite row per section instead of collating.

        Pass 1 (sequential): fix the canonical gene panel and the sorted
        label-category vocab across all silver files, writing the
        `gene_panel.pkl` / `label_categories.pkl` sidecars inference code
        expects (in_memory_dataset_blob.py:213-310). `process_anndata_batch`
        READS `self.gene_panel` / `self.label_categories`, so both must be
        set before pass 2. Each section's `.obs` is SPILLED to its own pickle
        rather than accumulated in RAM (the parent holds all obs at :241,
        which is itself tens of GB at 1.1e8 cells); a lightweight
        {batch_id: relpath} index is kept and restored as a lazy map.

        Pass 2 (per-section): run the INHERITED `process_anndata_batch`
        (returns ONE `Data`, in_memory_dataset_blob.py:585) and `append` it
        as a serialized SQLite row, releasing it immediately. Peak memory =
        one section.
        """
        meta = self._meta_dir
        obs_dir = meta / "obs"
        obs_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic order for reproducible builds.
        raw_files = sorted(self.raw_paths, key=lambda p: str(p))

        # ---------------- Pass 1: gene panel + label vocab (streamed) ------
        self.gene_panel = None
        self.label_categories = {ln: set() for ln in self.label_names}
        obs_index: dict = {}   # batch_id -> obs sidecar relpath

        for adata_batch_file in raw_files:
            # BACKED read: pass 1 only needs .var (gene panel), .obs (spilled
            # below) and .uns (batch id) — never .X. A full `sc.read` here
            # loaded every section into RAM in its entirety, and pass 2 then
            # read every file AGAIN via process_anndata_batch, so the build
            # paid two complete passes over the corpus. Section files reach
            # ~8.6 GB, so on 636 sections that is a large avoidable read.
            # `_panelsurvey.py` already established backed mode is enough for
            # exactly this metadata.
            adata_batch = ad.read_h5ad(adata_batch_file, backed="r")
            batch_id = self._derive_adata_batch_id(
                adata_batch=adata_batch, adata_batch_file=adata_batch_file,
            )

            obs_rel = f"obs/section_{batch_id:05d}.pkl"
            with open(meta / obs_rel, "wb") as f:
                pickle.dump(adata_batch.obs.copy(), f)
            obs_index[batch_id] = obs_rel

            if self.gene_panel is None:
                self.gene_panel = adata_batch.var
            else:
                if set(self.gene_panel.index) != set(adata_batch.var.index):
                    raise ValueError(
                        "All batches must share the same gene panel "
                        "(gene SETS differ between batches)."
                    )
                aligned_var = adata_batch.var.reindex(self.gene_panel.index)
                if not self.gene_panel.equals(aligned_var):
                    raise ValueError(
                        "All batches must share the same gene panel "
                        "(genes match but per-gene .var metadata differs)."
                    )

            for label_name, label_key in zip(self.label_names, self.label_keys):
                if label_key not in adata_batch.obs.columns:
                    print(f"WARNING: obs column '{label_key}' missing from "
                          f"{Path(adata_batch_file).name}; skipping "
                          f"y_{label_name} for this section.")
                    continue
                _vals = pd.Series(adata_batch.obs[label_key].unique()).dropna().tolist()
                self.label_categories[label_name].update(_vals)

            # Release the HDF5 handle; backed AnnData keeps the file open and
            # 636 open handles would exhaust the per-process limit.
            try:
                if adata_batch.file is not None:
                    adata_batch.file.close()
            except Exception:  # noqa: BLE001 - closing is best-effort
                pass
            del adata_batch

        for label_name in self.label_names:
            self.label_categories[label_name] = sorted(list(self.label_categories[label_name]))

        with open(meta / "label_categories.pkl", "wb") as f:
            pickle.dump(self.label_categories, f)
        with open(meta / "gene_panel.pkl", "wb") as f:
            pickle.dump(self.gene_panel, f)
        with open(meta / "obs_per_batch_id.pkl", "wb") as f:
            pickle.dump({"__lazy_obs_index__": obs_index}, f)

        # ---------------- Pass 2: one SQLite row per section ---------------
        section_ids: List[int] = []
        node_counts: List[int] = []
        batch_labels: List[str] = []
        for adata_batch_file in raw_files:
            print(f"Processing {adata_batch_file}...")
            # Inherited; pre_filter/pre_transform already applied inside it
            # (in_memory_dataset_blob.py:579-585).
            data_batch = self.process_anndata_batch(adata_batch_file)
            # Append as a serialized row. schema=object => serialize() is a
            # passthrough and the DB pickles the Data (list attrs survive).
            self.append(self.serialize(data_batch))
            section_ids.append(int(data_batch.adata_batch_id))
            # Cheap per-section facts recorded HERE so consumers never have
            # to deserialize a row just to learn them (see `manifest` below).
            node_counts.append(int(data_batch["x_cell_gene_counts"].shape[0]))
            batch_labels.append(_section_batch_label(data_batch))
            del data_batch   # never hold two sections at once

        # Row order == append order == sorted(raw_files) order.
        #
        # `node_counts` and `batch_labels` exist so that nothing downstream
        # needs a full-row fetch for metadata:
        #   - node_counts  -> KSectionBlockLoader.__len__ can size an epoch
        #     exactly without touching the DB. Without it, computing the
        #     length deserializes EVERY section (the whole corpus) to read
        #     one integer each.
        #   - batch_labels -> the label->dense-id map for per-cell batch ids
        #     must be built ONCE over all sections. Densifying per block
        #     would assign dense id 0 to a different section in every block.
        manifest = {
            "manifest_version": 2,
            "name": self.name,
            # Which container holds the rows. Recorded so a reopen knows what to
            # expect and so a blob built before this field defaults to sqlite.
            "container": self.backend,
            "num_sections": len(section_ids),
            "section_ids": section_ids,      # DB row idx -> adata_batch_id
            "node_counts": node_counts,      # DB row idx -> n_cells
            "batch_labels": batch_labels,    # DB row idx -> uns['batch'] value
            "feature_names": self.feature_names,
            "label_names": self.label_names,
        }
        with open(meta / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------ #
    # Metadata restore
    # ------------------------------------------------------------------ #
    def _load_sidecars(self) -> None:
        meta = self._meta_dir
        mpath = meta / "manifest.json"
        self.section_ids: List[int] = []
        # `node_counts` / `batch_labels` are manifest_version >= 2. Blobs
        # built before that stay usable: consumers fall back to deriving
        # them from the rows (see `node_counts` / `batch_labels` properties).
        self.node_counts: Optional[List[int]] = None
        self.batch_labels: Optional[List[str]] = None
        self.container: str = "sqlite"
        if mpath.exists():
            with open(mpath) as f:
                _manifest = json.load(f)
            self.section_ids = _manifest.get("section_ids", [])
            self.node_counts = _manifest.get("node_counts", None)
            self.batch_labels = _manifest.get("batch_labels", None)
            # Blobs built before the file container existed have no `container`
            # field and are sqlite by construction.
            self.container = _manifest.get("container", "sqlite")

        gp = meta / "gene_panel.pkl"
        if gp.exists():
            with open(gp, "rb") as f:
                self.gene_panel = pickle.load(f)

        lc = meta / "label_categories.pkl"
        if lc.exists():
            with open(lc, "rb") as f:
                self.label_categories = pickle.load(f)

        obs_pkl = meta / "obs_per_batch_id.pkl"
        if obs_pkl.exists():
            with open(obs_pkl, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict) and "__lazy_obs_index__" in loaded:
                self.obs_per_batch_id = _ObsLazyMap(meta, loaded["__lazy_obs_index__"])
            else:
                self.obs_per_batch_id = loaded
        else:
            self.obs_per_batch_id = {}

    # ------------------------------------------------------------------ #
    # Streaming consumption helpers
    # ------------------------------------------------------------------ #
    def get_node_counts(self) -> List[int]:
        """
        Cells per section, indexed by DB row — from the manifest when the
        blob was built with manifest_version >= 2, otherwise derived once by
        fetching each row and cached back onto the instance.

        The manifest path costs no I/O, which is the whole point: sizing an
        epoch used to deserialize the entire corpus (one full section per
        integer read).
        """
        if self.node_counts is None:
            print(
                "OnDiskDatasetBlob: manifest has no 'node_counts' (blob built "
                "before manifest_version 2) — deriving it by fetching every "
                "section once. Rebuild the blob to avoid this."
            )
            self.node_counts = [
                int(self.get(i)["x_cell_gene_counts"].shape[0])
                for i in range(len(self))
            ]
        return self.node_counts

    def get_batch_labels(self) -> List[str]:
        """
        Per-section batch label (`uns['batch']`), indexed by DB row — from
        the manifest when available, else derived once from the rows.

        Used to build ONE label->dense-id map across the whole corpus; see
        `KSectionBlockLoader` for why a per-block map would be wrong.
        """
        if self.batch_labels is None:
            print(
                "OnDiskDatasetBlob: manifest has no 'batch_labels' (blob built "
                "before manifest_version 2) — deriving it by fetching every "
                "section once. Rebuild the blob to avoid this."
            )
            self.batch_labels = [
                _section_batch_label(self.get(i)) for i in range(len(self))
            ]
        return self.batch_labels

    def batch_label_to_dense(self) -> dict:
        """
        Corpus-wide {batch label -> dense id}, ids assigned over the SORTED
        unique labels.

        Sorted-unique matches `build_batch_one_hot_from_obs`
        (initializers/initialize.py), so a section gets the same dense id
        whether it is loaded through the streaming or the in-memory path.
        """
        labels = sorted(set(self.get_batch_labels()))
        return {lbl: i for i, lbl in enumerate(labels)}

    def iter_sections(self) -> Iterator[Tuple[int, Data]]:
        """Yield (row_idx, section Data) one at a time — never more than one
        section resident. Drives a per-section NeighborLoader in the training
        loop. Fetch is `get(idx)` == deserialize(db.get(idx))."""
        for idx in range(len(self)):
            yield idx, self.get(idx)

    @staticmethod
    def sampling_view(section: Data) -> Data:
        """
        Return a shallow copy of a section with python-list node attrs
        (`cell_id`, `obs_batch`) removed, so `NeighborLoader` can sample
        subgraphs. Verified necessary on PyG 2.6.1: NeighborLoader raises
        "invalid feature tensor type (got 'list')" otherwise. The removed
        metadata stays available on the original section (recover by row
        index via `get(idx)` or the obs sidecars).
        """
        view = copy.copy(section)          # shallow: shares tensors, own attr dict
        for attr in _NON_TENSOR_NODE_ATTRS:
            if attr in view:
                del view[attr]
        return view

    def to_concatenated(self) -> Data:
        """
        ESCAPE HATCH — collate all sections into a single in-memory `Data`,
        reproducing what `InMemoryDatasetBlob` hands `InMemoryDataModule`.
        Use ONLY for datasets small enough to fit in RAM (where streaming
        isn't needed); defeats the purpose on hst_corpus_110m. Lets existing
        training code run unchanged on small blobs built with this class.
        """
        from torch_geometric.data import Batch
        return Batch.from_data_list([self.get(i) for i in range(len(self))])


class _ObsLazyMap:
    """
    Dict-like, lazy-loading view over per-section obs DataFrames spilled to
    disk during `process()`. `obs_map[batch_id]` reads and caches that
    section's obs pickle on first access, so inference-time obs write-back
    (`inference_data_dict_to_adata`) works exactly as with the eager
    `obs_per_batch_id` dict, without holding every section's obs in RAM.
    """

    def __init__(self, root: Path, index: dict):
        self._root = Path(root)
        self._index = index               # batch_id -> relpath
        self._cache: dict = {}

    def __contains__(self, key) -> bool:
        return key in self._index

    def __iter__(self):
        return iter(self._index)

    def keys(self):
        return self._index.keys()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, key):
        if key not in self._cache:
            with open(self._root / self._index[key], "rb") as f:
                self._cache[key] = pickle.load(f)
        return self._cache[key]

    def get(self, key, default=None):
        return self[key] if key in self._index else default


class KSectionBlockLoader:
    """
    Streaming, section-mixing loader for `OnDiskDatasetBlob` (Direction 1).

    THE PROBLEM IT SOLVES
    ---------------------
    The in-memory pipeline concatenates EVERY section into one resident graph
    and lets a NeighborLoader draw seed cells from all sections at once — so a
    mini-batch mixes cells from many tissue sections, which is what feeds the
    batch-integration mechanisms (adversarial GRL / decoder covariate / MMD).
    Pure per-section streaming (iter_sections + one loader per section) fixes
    memory but DESTROYS that mixing: a mini-batch would then contain a single
    section, giving no cross-batch signal.

    HOW IT WORKS
    ------------
    Load K sections at a time ("a block"), concatenate them into one block
    graph with `Batch.from_data_list` — which OFFSETS each section's
    `edge_index` by its node count, so NO cross-section edges are created and
    every section stays a disconnected component. Run a NeighborLoader over
    the block:
      - SEED cells are drawn across all K sections => a mini-batch MIXES
        sections (the property we want), and
      - each seed's sampled NEIGHBORHOOD stays inside its own section
        (because there are no cross-section edges).
    MEMORY. Residency is bounded by a small multiple of K sections, not by the
    corpus. Budget ~2xK, not K: `Batch.from_data_list` allocates the combined
    block while the K source views are still referenced, so both exist at the
    moment of concatenation. With `prefetch=True` one further block is being
    prepared while the current one is consumed. K trades mixing breadth against
    memory: K=1 == per-section (no mixing); K=all == the in-memory blob (full
    mixing, full memory).

    Each block is reshuffled every epoch (section->block assignment changes),
    so over epochs a given section co-occurs with many others.

    Per-cell section provenance is exposed as `section_row` (long, the dataset
    row index of the originating section) so the model / loss can tell which
    tissue batch each cell came from even after subgraph sampling. Note the
    section id in a built section `Data` is the SCALAR `adata_batch_id`
    (in_memory_dataset_blob.py:565, one integer per section, set from
    uns['batch']), not a per-cell vector; the per-cell batch vector the
    in-memory DataModule consumes is `adata_batch_ids`
    (in_memory_datamodule.py:378), with an optional per-cell `adata_batch_ids_raw`
    (in_memory_datamodule.py:397-404, only forwarded when present via getattr).
    `section_row` here is the streaming analogue — a per-cell long we set from
    the block's dataset row index — and is also handy for asserting that mixing
    happened.
    """

    def __init__(
        self,
        dataset: "OnDiskDatasetBlob",
        edge_index_name: str,
        sections_per_block: int = 4,
        batch_size: int = 256,
        num_neighbors: List[int] = (8,),
        shuffle: bool = True,
        seed: int = 0,
        input_mask_attr: Optional[str] = None,
        section_transform: Optional[Callable] = None,
        batch_label_to_dense: Optional[dict] = None,
        unknown_batch_label_dense_id: int = 0,
        prefetch: bool = True,
        loader_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        dataset : OnDiskDatasetBlob
            The on-disk blob to stream sections from.
        edge_index_name : str
            Which spatial graph to sample on, e.g. "spatial_n_neighs_8".
            Sections store one `edge_index_<edge_index_name>` per k in
            `n_neighs_list`; this picks the k the variant trains on.
        sections_per_block : int
            K — number of sections co-resident and mixed per block. Set to
            fit K sections in the memory budget (see the size estimate:
            ~1-6 GB/section for hst_corpus).
        batch_size : int
            NeighborLoader seed-node batch size (mini-batch size).
        num_neighbors : list[int]
            Neighbors sampled per hop (train config; val/test typically [-1]).
        shuffle : bool
            Shuffle section->block assignment each epoch AND seeds within a
            block. Set False for val/test/predict for reproducible order.
        input_mask_attr : str | None
            Node-level boolean attr (e.g. "train_mask") restricting which
            cells may be SEEDS. None => every cell is a valid seed.
        batch_label_to_dense : dict | None
            Corpus-wide {batch label -> dense id} used to stamp each cell with
            `adata_batch_ids`, which the decoder covariate / adversarial head /
            FiLM conditioning all require (models/vqniche_dual.py:492, :659
            raise without it). Build it once over ALL sections —
            `OnDiskDatasetBlob.batch_label_to_dense()` does that from the
            manifest. Passing None skips stamping, which is only correct when
            every batch-correction mechanism is disabled.
        unknown_batch_label_dense_id : int
            Dense id for labels absent from the map (predict-time novel
            sections). Those cells are flagged in
            `adata_batch_ids_unseen_mask` so the model can substitute a mean
            batch embedding. Matches `build_batch_one_hot_from_obs`.
        section_transform : Callable | None
            PyG transform (or Compose) applied to EACH section Data BEFORE
            list-strip + concatenation. This is where the in-memory pipeline's
            per-Data transforms run in the streaming path: split transforms
            (RandomNodeSplit / SpatialBatchSplit -> set train/val/test_mask)
            and SetExperimentDataKeys (x_cell_gene_counts -> data.x, y_* ->
            data.y, edge_index_<name> -> data.edge_index, num_features).
            Applied per section so masks/keys are computed within each
            section, exactly as the in-memory path applies them to each
            section before collation. When it already sets `data.x` /
            `data.edge_index`, the fallback below is skipped.
        prefetch : bool
            Build the next block on a background thread while the current one
            is being consumed, so the GPU does not idle at block boundaries.
            Costs one extra resident block (see the memory note in the class
            docstring). Set False to debug or to minimise memory.
        loader_kwargs : dict | None
            Extra kwargs forwarded to NeighborLoader. NOTE `num_workers` here
            is usually counter-productive: a NeighborLoader is constructed per
            block, so workers are spawned and torn down for EVERY block (159
            per epoch at K=4 over 636 sections) and `persistent_workers` cannot
            help. `prefetch=True` hides the same disk cost without that
            churn — prefer it and leave `num_workers=0`.
        """
        self.dataset = dataset
        self.edge_index_name = edge_index_name
        self.sections_per_block = int(sections_per_block)
        self.batch_size = int(batch_size)
        self.num_neighbors = list(num_neighbors)
        self.shuffle = shuffle
        self.seed = seed
        self.input_mask_attr = input_mask_attr
        # Fail fast on transforms that cannot be applied per section.
        _reject_global_scope_transforms(section_transform)
        self.section_transform = section_transform
        # Corpus-wide {batch label -> dense id}. MUST span every section, not
        # just this block's — see `_stamp_batch_ids`.
        self.batch_label_to_dense = batch_label_to_dense
        self.unknown_batch_label_dense_id = int(unknown_batch_label_dense_id)
        self.prefetch = bool(prefetch)
        # Per-thread SQLite read handles for prefetching; see `_fetch_section`.
        import threading as _threading
        self._thread_local = _threading.local()
        self.loader_kwargs = dict(loader_kwargs or {})
        self._epoch = 0
        # {mask attr -> per-section seed counts}. Populated on first
        # use by `_seed_counts`; see `__len__` for why it is needed.
        self._seed_count_cache: dict = {}

    # -- block assignment -------------------------------------------------- #
    def _block_row_lists(self) -> List[List[int]]:
        order = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self._epoch).shuffle(order)
        K = self.sections_per_block
        return [order[i:i + K] for i in range(0, len(order), K)]

    @staticmethod
    def _view_num_nodes(view: Data) -> int:
        """
        Cell count of a section view: `x` once `SetExperimentDataKeys` has run,
        otherwise the raw counts matrix. Needed in two places (per-cell batch
        ids, and `num_nodes`), so kept in one place.
        """
        if "x" in view and view.x is not None:
            return int(view.x.shape[0])
        return int(view["x_cell_gene_counts"].shape[0])

    # -- per-section view prepared for concatenation ----------------------- #
    def _block_view(self, section: Data, row: int) -> Data:
        """
        Prepare ONE section for concatenation: (optionally) apply the
        per-section transform, strip non-tensor list attrs, ensure a
        canonical `edge_index`, drop the remaining `edge_index_*` variants,
        tag per-cell provenance, and set `num_nodes`. The result is safe to
        pass to `Batch.from_data_list` then `NeighborLoader`.
        """
        view = copy.copy(section)

        # 0) Set num_nodes up front. Sections store features under
        #    `x_cell_gene_counts` (not `x`) until SetExperimentDataKeys runs,
        #    so PyG can't infer num_nodes — split transforms (RandomNodeSplit)
        #    assert on it. Seed it from the raw counts before any transform.
        if "x_cell_gene_counts" in view and view["x_cell_gene_counts"] is not None:
            view.num_nodes = int(view["x_cell_gene_counts"].shape[0])

        # 1) Per-section transform (split masks + SetExperimentDataKeys). Runs
        #    on the section BEFORE concatenation so masks/keys are within-
        #    section, matching the in-memory path.
        if self.section_transform is not None:
            view = self.section_transform(view)

        # 2) Strip python-list node attrs (NeighborLoader rejects them).
        for attr in _NON_TENSOR_NODE_ATTRS:
            if attr in view:
                del view[attr]

        # 3) Ensure a canonical `edge_index`. If the transform already set it
        #    (SetExperimentDataKeys does), respect that; else promote the
        #    chosen edge_index_<name>.
        if "edge_index" not in view or view.edge_index is None:
            ekey = f"edge_index_{self.edge_index_name}"
            if ekey not in view:
                raise KeyError(
                    f"Section (row {row}) has no '{ekey}' and no transform set "
                    f"`edge_index`. Available edge_index keys: "
                    f"{[k for k in view.keys() if k.startswith('edge_index_')]}. "
                    f"Build the blob with this k in n_neighs_list, or pass a "
                    f"section_transform that sets edge_index."
                )
            view.edge_index = view[ekey]

        # 4) Drop leftover edge_index_* variants: PyG offsets ANY attr whose
        #    name contains 'edge_index' during from_data_list, so carrying the
        #    unused graphs wastes memory and confuses offsetting.
        for k in [kk for kk in list(view.keys())
                  if kk.startswith("edge_index_") and kk != "edge_index"]:
            del view[k]

        # 5) num_nodes: prefer data.x (set by transform), else raw counts.
        n = self._view_num_nodes(view)

        # 5b) Drop per-section SCALAR / string metadata (adata_batch_id,
        #     dataset_id, species, tissue, ...). Batch.from_data_list would
        #     stack a scalar into a length-K vector, which then breaks PyG's
        #     node/edge attribute inference during NeighborLoader collate
        #     ("num_nodes != num_edges" comparison on a tensor). Anything the
        #     model needs at cell level must be a per-CELL tensor of length n
        #     (e.g. `adata_batch_ids`, set in step 1b above); keep only
        #     tensors whose first dim is n (node attrs), the 2-row edge_index,
        #     and num_nodes. Everything else is dropped from the streaming
        #     batch (recover section-level metadata by row via `get(idx)`).
        for k in list(view.keys()):
            if k in ("edge_index", "num_nodes"):
                continue
            v = view[k]
            if torch.is_tensor(v):
                # keep node-level tensors (first dim == n); drop others
                if v.dim() >= 1 and v.shape[0] == n:
                    continue
                del view[k]
            else:
                # scalars, strings, None, etc.
                del view[k]

        view.num_nodes = n

        # 6) Per-cell section provenance (survives subgraph sampling).
        view.section_row = torch.full((n,), int(row), dtype=torch.long)
        return view

    # -- per-cell batch identity (stamped AFTER collation) ----------------- #
    def _stamp_batch_ids(self, block: Data, rows: List[int],
                         sizes: List[int]) -> None:
        """
        Stamp `adata_batch_ids` / `adata_batch_ids_unseen_mask` onto an
        already-collated block.

        These are what the model's batch-correction machinery consumes; the
        decoder covariate and the adversarial head both raise without them
        (models/vqniche_dual.py:492-497, :659-666). The in-memory path builds
        them in `initialize_databatch` (initializers/initialize.py:466-478);
        streaming never calls that, so we do the equivalent here.

        WHY AFTER `Batch.from_data_list`, not on each section view:
        PyG's `Data.__inc__` offsets any attribute whose KEY CONTAINS "batch"
        by `int(value.max()) + 1` per element of the list, so that per-graph
        `batch` vectors concatenate into a global assignment. `adata_batch_ids`
        matches that substring rule, so setting it per section and then
        collating silently shifts the ids — three sections stamped 0/1/2 come
        out as 0/2/5. Verified on PyG 2.6.1. Stamping after collation avoids
        `__inc__` entirely, and matches the in-memory ordering (ids are
        assigned to the collated object there too).

        Labels come from the manifest (`get_batch_labels`), so this costs no
        extra I/O.
        """
        if self.batch_label_to_dense is None:
            return
        labels = self.dataset.get_batch_labels()
        ids, unseen = [], []
        for row, n in zip(rows, sizes, strict=True):
            label = str(labels[row])
            known = label in self.batch_label_to_dense
            dense = (self.batch_label_to_dense[label] if known
                     else self.unknown_batch_label_dense_id)
            ids.append(torch.full((n,), int(dense), dtype=torch.long))
            unseen.append(torch.full((n,), not known, dtype=torch.bool))
        block.adata_batch_ids = torch.cat(ids)
        # True for sections whose label was absent from the (train-time) map,
        # i.e. predict-time novel batches: the model substitutes a mean batch
        # embedding rather than an arbitrary reference batch.
        block.adata_batch_ids_unseen_mask = torch.cat(unseen)

    # -- section fetch (thread-aware) -------------------------------------- #
    def _fetch_section(self, row: int) -> Data:
        """
        Fetch one section by DB row.

        Every read goes through here so prefetching has a single place to be
        thread-safe. `sqlite3` connections are bound to the thread that
        created them (`check_same_thread=True` by default, and PyG's
        `SQLiteDatabase.connect()` does not override it), so a prefetch thread
        calling the shared handle raises

            sqlite3.ProgrammingError: SQLite objects created in a thread can
            only be used in that same thread

        Rebinding the shared handle is not an option — the main thread also
        reads (`_seed_counts`, `to_concatenated`), so whichever thread
        reconnected last would break the other. Instead each non-main thread
        opens its OWN read handle to the same file, cached in thread-local
        storage. Concurrent readers on one SQLite file are fine.

        `deserialize` is the dataset's, so rows decode identically either way.
        """
        import threading

        # The 'file' container has no connection and therefore no thread affinity, so
        # every thread can read directly. This is one of the reasons it was preferred
        # over another connection-based backend.
        if (threading.current_thread() is threading.main_thread()
                or getattr(self.dataset, "backend", None) != "sqlite"):
            return self.dataset.get(row)

        db = getattr(self._thread_local, "db", None)
        if db is None:
            from torch_geometric.data.database import SQLiteDatabase

            db = SQLiteDatabase(
                path=self.dataset.db.path,
                name=self.dataset.db.name,
                schema=self.dataset.schema,
            )
            self._thread_local.db = db
        return self.dataset.deserialize(db.get(row))

    # -- iteration --------------------------------------------------------- #
    def _build_block(self, block_rows: List[int]):
        """
        Fetch K sections, prepare and concatenate them into one block graph.

        This is the expensive step (disk read + per-section transform + a full
        concatenating copy), which is why `__iter__` can run it ahead of time on
        a background thread.
        """
        from torch_geometric.data import Batch

        views = [self._block_view(self._fetch_section(r), r) for r in block_rows]
        sizes = [int(v.num_nodes) for v in views]
        block = Batch.from_data_list(views)              # edges offset per section
        # Must come AFTER collation — see `_stamp_batch_ids` for why.
        self._stamp_batch_ids(block, block_rows, sizes)
        return block

    def _blocks(self) -> Iterator:
        """
        Yield prepared blocks, optionally reading the next one ahead.

        With `prefetch=False` this is a plain synchronous generator. With
        `prefetch=True` a daemon thread builds block n+1 while the caller is
        still training on block n, so the GPU no longer idles at every block
        boundary — previously each boundary stalled for K x (deserialize +
        transform), once per block per epoch.

        A thread (not a process) is sufficient: the cost is HDF5/SQLite reads
        and tensor copies, which release the GIL. The queue holds a single
        block, so steady-state residency is 2 blocks rather than 1 — budget K
        accordingly (and see the 2xK note in the class docstring).
        """
        row_lists = self._block_row_lists()
        if not self.prefetch:
            for block_rows in row_lists:
                yield self._build_block(block_rows)
            return

        import queue
        import threading

        q: "queue.Queue" = queue.Queue(maxsize=1)
        _DONE = object()

        def producer():
            try:
                for block_rows in row_lists:
                    q.put(self._build_block(block_rows))
            except BaseException as exc:  # noqa: BLE001 - re-raised in consumer
                q.put(exc)
            else:
                q.put(_DONE)
            finally:
                # Release this thread's private read handle.
                db = getattr(self._thread_local, "db", None)
                if db is not None:
                    try:
                        db.close()
                    except Exception:  # noqa: BLE001 - best effort
                        pass
                    self._thread_local.db = None

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                # Surface producer failures on the consumer's stack rather
                # than losing them in a dead thread.
                raise item
            yield item

    def __iter__(self):
        from torch_geometric.loader import NeighborLoader

        for block in self._blocks():

            input_nodes = None
            if self.input_mask_attr is not None and self.input_mask_attr in block:
                input_nodes = block[self.input_mask_attr]

            nl = NeighborLoader(
                block,
                num_neighbors=self.num_neighbors,
                batch_size=self.batch_size,
                input_nodes=input_nodes,
                shuffle=self.shuffle,
                **self.loader_kwargs,
            )
            for mini_batch in nl:
                yield mini_batch

            del block                                     # release the K sections
        self._epoch += 1

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so block reshuffling is deterministic per epoch
        (call from the training loop if not relying on the internal counter)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        """
        EXACT number of mini-batches the next `__iter__` will yield.

        This must be exact, because Lightning takes it as `num_training_batches`
        and stops the epoch there (pytorch_lightning/loops/fit_loop.py:253) —
        any under-count silently drops the tail of every epoch.

        The count is per-BLOCK, not global. Each block builds its own
        NeighborLoader, so each block contributes its own partial final batch:

            correct  = sum over blocks of ceil(block_seeds / batch_size)
            previous = ceil(sum of all seeds / batch_size)

        A sum of ceilings is never smaller than the ceiling of the sum, so the
        old formula could only ever under-count — by up to (n_blocks - 1)
        batches per epoch, exact only when K covers every section in one block.

        Uses the SAME block partition `__iter__` will use (both call
        `_block_row_lists()`, which is deterministic for the current epoch), so
        the two cannot disagree.

        Cost: zero DB reads in the common case — per-section cell counts come
        from the manifest. When `input_mask_attr` restricts seeds (train/val
        splits) the mask is produced by `section_transform` at load time and
        cannot be known from the manifest, so seed counts are computed once by
        streaming the sections and then cached per mask attribute.
        """
        seeds = self._seed_counts()
        return sum(
            math.ceil(sum(seeds[r] for r in rows) / self.batch_size)
            for rows in self._block_row_lists()
            if sum(seeds[r] for r in rows) > 0
        )

    def _seed_counts(self) -> List[int]:
        """
        Per-section count of cells eligible to be SEED nodes, indexed by DB row.

        No `input_mask_attr` -> every cell is a seed, so this is just the
        manifest's node counts and costs no I/O.

        With a mask -> the split masks come from `section_transform`, which only
        exists at load time, so this streams every section once (applying the
        transform) and caches the result on the instance, keyed by mask
        attribute. One pass per loader, not one per `len()` call as before.
        """
        if self.input_mask_attr is None:
            return self.dataset.get_node_counts()

        cached = self._seed_count_cache.get(self.input_mask_attr)
        if cached is not None:
            return cached

        print(
            f"KSectionBlockLoader: counting '{self.input_mask_attr}' seeds by "
            f"streaming {len(self.dataset)} sections once (split masks are "
            f"produced at load time, so they are not in the manifest). "
            f"Cached for the rest of this loader's life."
        )
        counts: List[int] = []
        for row in range(len(self.dataset)):
            view = self._block_view(self._fetch_section(row), row)
            if self.input_mask_attr in view:
                counts.append(int(view[self.input_mask_attr].sum()))
            else:
                # Transform produced no such mask -> treat every cell as a
                # seed, matching NeighborLoader's behaviour for input_nodes=None.
                counts.append(int(view.num_nodes))
        self._seed_count_cache[self.input_mask_attr] = counts
        return counts


try:
    import pytorch_lightning as _pl
    _LightningDataModuleBase = _pl.LightningDataModule
except Exception:  # lightning not importable in this context
    try:
        import lightning.pytorch as _pl  # newer namespace
        _LightningDataModuleBase = _pl.LightningDataModule
    except Exception:
        _LightningDataModuleBase = object


class OnDiskStreamingDataModule(_LightningDataModuleBase):
    """
    Lightning DataModule that streams an `OnDiskDatasetBlob` via
    `KSectionBlockLoader`, as a drop-in alternative to `InMemoryDataModule`
    for corpora too large to collate into one resident graph.

    Contract parity with `InMemoryDataModule`
    ------------------------------------------
    - Yields the SAME mini-batch shape the model consumes: after the
      per-section `section_transform` (split transforms + SetExperimentDataKeys)
      each mini-batch carries `batch.x`, `batch.y`, `batch.edge_index`,
      `batch.batch_size`, plus the streaming provenance `batch.section_row`.
    - Per-cell `adata_batch_ids` (and `adata_batch_ids_unseen_mask`) are
      stamped by `KSectionBlockLoader._block_view` from the section's batch
      label, using a corpus-wide label->dense map. These are what the decoder
      covariate, adversarial head and FiLM conditioning consume; the model
      raises without them (models/vqniche_dual.py:492-497, :659-666). An
      earlier version of this docstring claimed a `section_transform` should
      supply them — no such transform exists, the in-memory path builds them
      in `initialize_databatch`, which streaming never calls.
    - TRAIN uses the configured `num_neighbors` and mixes K sections per block
      (section-mixing => batch-integration signal). VAL/TEST/PREDICT override
      to `num_neighbors=[-1]` (full neighborhood) and `shuffle=False` for
      reproducible embeddings — mirroring `InMemoryDataModule`'s val/test/
      predict policy (in_memory_datamodule.py: val/test set num_neighbors=[-1]).

    Memory
    ------
    Independent of corpus size, but budget ~2xK sections
    (`sections_per_block`) plus the active NeighborLoader sub-batch — the
    concatenating copy coexists with the K source sections, and `prefetch`
    keeps one more block in flight. Sizing from K alone under-budgets by about
    half.

    Notes
    -----
    - The split masks (`train_mask`/`val_mask`/`test_mask`) are produced by the
      `section_transform` you pass, applied PER SECTION. RandomNodeSplit gives
      an in-section cell-level split; SpatialBatchSplit gives whole-section /
      region holdout — both work unchanged because they operate on one Data.
    - `input_train_nodes` semantics are reproduced via `input_mask_attr=
      "train_mask"` on the train loader (seeds restricted to train cells).
    """

    def __init__(
        self,
        dataset: "OnDiskDatasetBlob",
        edge_index_name: str,
        sections_per_block: int = 4,
        batch_size: int = 256,
        num_neighbors: List[int] = (8,),
        section_transform: Optional[Callable] = None,
        val_num_neighbors: Optional[List[int]] = None,
        num_workers: int = 0,
        seed: int = 0,
        batch_label_to_dense: Optional[dict] = None,
        unknown_batch_label_dense_id: int = 0,
        prefetch: bool = True,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.edge_index_name = edge_index_name
        self.sections_per_block = int(sections_per_block)
        self.batch_size = int(batch_size)
        self.num_neighbors = list(num_neighbors)
        # val/test/predict: full neighborhood unless caller overrides.
        self.val_num_neighbors = list(val_num_neighbors) if val_num_neighbors is not None else [-1]
        self.section_transform = section_transform
        self.num_workers = int(num_workers)
        self.seed = seed
        # Derive the corpus-wide batch map from the manifest unless the caller
        # supplied one (predict time must reuse the TRAIN-time map, otherwise
        # dense ids shift and the decoder covariate embedding is indexed out of
        # range). Same reason `initialize_databatch` takes it as an argument.
        self.batch_label_to_dense = (
            batch_label_to_dense if batch_label_to_dense is not None
            else dataset.batch_label_to_dense()
        )
        self.unknown_batch_label_dense_id = int(unknown_batch_label_dense_id)
        self.prefetch = bool(prefetch)

    def _make_loader(self, *, split: str) -> KSectionBlockLoader:
        is_train = split == "train"
        mask_attr = {
            "train": "train_mask",
            "val": "val_mask",
            "test": "test_mask",
            "predict": None,      # predict over all cells
        }[split]
        return KSectionBlockLoader(
            dataset=self.dataset,
            edge_index_name=self.edge_index_name,
            sections_per_block=self.sections_per_block,
            batch_size=self.batch_size,
            num_neighbors=self.num_neighbors if is_train else self.val_num_neighbors,
            shuffle=is_train,
            seed=self.seed,
            input_mask_attr=mask_attr,
            section_transform=self.section_transform,
            batch_label_to_dense=self.batch_label_to_dense,
            unknown_batch_label_dense_id=self.unknown_batch_label_dense_id,
            prefetch=self.prefetch,
            loader_kwargs={"num_workers": self.num_workers},
        )

    def train_dataloader(self):
        return self._make_loader(split="train")

    def val_dataloader(self):
        return self._make_loader(split="val")

    def test_dataloader(self):
        return self._make_loader(split="test")

    def predict_dataloader(self):
        return self._make_loader(split="predict")
