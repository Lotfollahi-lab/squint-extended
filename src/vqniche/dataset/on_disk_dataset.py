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
import gc
import json
import math
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import Callable, List, Optional, Iterator, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, OnDiskDataset

from .file_database import FileDatabase
from .in_memory_dataset_blob import InMemoryDatasetBlob


# Node-level attributes that are python lists / non-tensors and therefore
# must be removed from a section before NeighborLoader subgraph sampling.
# (Set at in_memory_dataset_blob.py:522 (cell_id) and :556 (obs_batch).)
_NON_TENSOR_NODE_ATTRS = ("cell_id", "obs_batch")


# Transform-scope guard. Lives in `transform_scope` because BOTH backends apply their
# composed transform per section, so the in-memory path
# (initializers/initialize.py) needs the identical check —
# see that module for the full rationale. Re-exported here so
# `KSectionBlockLoader` and existing importers are unaffected.
from .transform_scope import (  # noqa: E402  (kept next to its usage)
    _GLOBAL_SCOPE_TRANSFORMS,
    _iter_transforms,
    _reject_global_scope_transforms,
)

__all__ = [
    "OnDiskDatasetBlob",
    "KSectionBlockLoader",
    "OnDiskStreamingDataModule",
    "_GLOBAL_SCOPE_TRANSFORMS",
    "_iter_transforms",
    "_reject_global_scope_transforms",
]


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


def _rel_section_name(path, raw_dir) -> str:
    """
    A section's corpus-unique name: its path relative to the silver root,
    i.e. "subdir/file.h5ad". Falls back to "<parent>/<name>" when the path is
    not under `raw_dir`, which keeps the name stable for the tmp-dir layouts
    the tests build.
    """
    p = Path(path)
    try:
        return str(p.relative_to(Path(raw_dir)))
    except ValueError:
        return f"{p.parent.name}/{p.name}" if p.parent.name else p.name


def _warn_on_batch_label_collisions(batch_labels, dataset_ids, batch_key) -> None:
    """
    Shout when one batch label spans several datasets.

    A label that names 80 unrelated experiments silently destroys the decoder
    covariate and the GRL adversary, and it is invisible in every metric — the
    run completes and the numbers look plausible. This is the guard that stops
    it recurring, and it fires regardless of `batch_key` so that even a
    deliberate legacy build says so out loud.
    """
    if not batch_labels or not any(d is not None for d in dataset_ids):
        return
    spans: dict = {}
    for lbl, dsid in zip(batch_labels, dataset_ids, strict=True):
        if dsid is not None:
            spans.setdefault(lbl, set()).add(dsid)
    bad = {k: v for k, v in spans.items() if len(v) > 1}
    if not bad:
        return
    n_sec = sum(1 for lbl in batch_labels if len(spans.get(lbl, ())) > 1)
    worst = sorted(bad.items(), key=lambda kv: -len(kv[1]))[:5]
    print(
        f"\n*** WARNING: batch label collision across datasets ***\n"
        f"  {len(bad)} of {len(spans)} batch labels are used by more than one "
        f"dataset, covering {n_sec}/{len(batch_labels)} sections.\n"
        + "".join(f"    {k!r} spans {len(v)} datasets\n" for k, v in worst)
        + f"  batch_key={batch_key!r}. Every section sharing a label is treated "
        f"as ONE batch by the\n"
        f"  decoder covariate, FiLM conditioning and the adversarial head, and "
        f"held-out sections\n"
        f"  carrying a seen label will not register as unseen. Rebuild with "
        f"batch_key='dataset_batch'\n"
        f"  unless the collision is genuinely intended.\n",
        flush=True,
    )


def _report_vocab_drops(drops, vocab_size) -> None:
    """
    Per-section accounting for the vocabulary cap.

    Reported rather than silent BY DESIGN: dropping stored columns without
    saying so is precisely what made a gene filter unacceptable, so a build
    that discards data has to state how much and from where.
    """
    tot_cols = sum(d["dropped_cols"] for d in drops)
    worst = sorted(drops, key=lambda d: -d["dropped_frac_counts"])[:10]
    print(
        f"\nVocabulary cap: {len(drops)} of the built sections lost columns "
        f"(vocabulary {vocab_size} genes; {tot_cols} column-slots dropped in "
        f"total).\n  Worst-hit by FRACTION OF COUNTS lost:", flush=True,
    )
    for d in worst:
        print(
            f"    {d['rel']:44} {d['dropped_cols']:>6}/{d['native_cols']:<6} "
            f"cols  {d['dropped_frac_counts']:>7.2%} of counts", flush=True,
        )


def _section_dataset_id(section) -> Optional[str]:
    """
    The section's `uns['dataset_id']`, stamped by `process_anndata_batch`
    (in_memory_dataset_blob.py:530, defaulting to the blob name when absent).
    """
    dsid = section.get("dataset_id", None) if hasattr(section, "get") else None
    if dsid is None:
        dsid = getattr(section, "dataset_id", None)
    if dsid is None:
        return None
    # Stored as a plain string, but a list-valued uns round-trips as a list.
    if isinstance(dsid, (list, tuple)):
        dsid = dsid[0] if len(dsid) else None
    return None if dsid is None else str(dsid)


def _section_batch_identity(section, batch_key: str = "batch") -> str:
    """
    The label the decoder covariate, FiLM conditioning and GRL adversary treat
    as "which batch is this cell from" — see `OnDiskDatasetBlob.__init__` for
    why `batch_key` exists and why the composite is required at corpus scale.

    Falls back to the bare batch label when `dataset_id` is absent, rather than
    fabricating a key: a blob whose sections predate the `dataset_id` stamp
    keeps its old identities instead of silently collapsing them all onto a
    single "None_batchN".
    """
    label = _section_batch_label(section)
    if batch_key != "dataset_batch":
        return label
    dsid = _section_dataset_id(section)
    return label if dsid is None else f"{dsid}_{label}"


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
        cross_panel: bool = False,
        exclude_sections: Optional[List[str]] = None,
        include_sections: Optional[List[str]] = None,
        output_suffix: str = "",
        resume: bool = False,
        min_panels_per_gene: int = 1,
        gene_vocab: Optional[List[str]] = None,
        batch_key: str = "batch",
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

        # ---- cross-panel (union + mask) --------------------------------------
        # OFF by default, so every existing blob, variant and test keeps the
        # single-panel behaviour: pass 1 hard-raises unless all sections share
        # an identical `.var`.
        #
        # ON, sections are stored at their OWN gene width with a `gene_ids`
        # index into a corpus-wide vocabulary, and the union is assembled per
        # BLOCK at load time. Storing at a padded global width instead would be
        # simpler but is not affordable: the corpus is ~1.21 TB dense at the
        # section-weighted mean width of 2,752 genes, 4.21 TB padded to the
        # 9,571-gene union, 8.33 TB padded to the full 18,937.
        #
        # `exclude_sections` drops sections by file stem BEFORE the vocabulary
        # is computed. That ordering is the point: the vocabulary is the union
        # over the sections actually built, so a built section can never carry a
        # gene outside it. Filtering GENES instead would silently drop columns
        # from sections that measured them -- excluding `chp60-1b_1p`'s ~9,366
        # exclusive genes while still building that section would discard ~49%
        # of its data with no error. This is also the knob for growing the
        # vocabulary later: build with the section included and V goes from
        # 9,571 to 18,937 with no code change.
        self.cross_panel = bool(cross_panel)
        self.exclude_sections = set(exclude_sections or ())
        # Restrict the build to these sections, applied BEFORE the vocabulary
        # like `exclude_sections` and matched the same way (rel path, or an
        # unambiguous stem). Exists so a representative SUBSET of a corpus can
        # be built in minutes to validate a pipeline change before committing
        # hours to the full thing -- three build-stopping bugs in a row were
        # each cheaper to find on 21 sections than on 636.
        # Suffixes the OUTPUT directory only, never `self.name` -- `raw_dir`
        # is `silver / self.name`, so renaming the blob would look for silver
        # data that does not exist. Lets a subset build occupy its own gold
        # directory while reading the same silver source.
        self.output_suffix = str(output_suffix or "")
        self.include_sections = set(include_sections or ())
        if self.include_sections and self.exclude_sections:
            raise ValueError(
                "Pass include_sections or exclude_sections, not both: the "
                "intersection is silent and easy to get wrong."
            )
        self.gene_vocab = None            # pd.Index of gene names, len V
        self._gene_ids_per_batch: dict = {}   # batch_id -> np.ndarray[G_sec]
        # batch_id -> column positions KEPT from the section's native width.
        # Empty (and unused) unless the vocabulary drops columns.
        self._gene_cols_per_batch: dict = {}

        # ---- vocabulary policy ----------------------------------------------
        # `min_panels_per_gene` caps the vocabulary at genes corroborated by at
        # least N distinct PANELS (a panel is a distinct gene SET -- 46 of them
        # across the corpus -- not a section; a gene in 150 sections that all
        # share one panel is still a one-panel gene).
        #
        # Measured by `_vocabpolicy.py` over all 636 sections: threshold 1 gives
        # 18,937 genes, 2 gives 9,574, 3 gives 6,581. Threshold 2 is a
        # `chp60-1b_1p` operation and essentially nothing else -- of the 9,363
        # genes it drops, 9,353 are chp60's, and the other 10 come from four
        # datasets losing 1-4 genes each. `chr78-11b_1p` (the unseen-assay rung,
        # and the corpus's narrowest panel at 169 genes) retains 100%.
        #
        # Why a threshold rather than excluding the wide section outright: it
        # keeps chp60's 48,934 cells and its ~9,540 corroborated genes, while
        # avoiding a V that doubles -- and therefore a 4x block-memory spike and
        # 49% of the model's gene rows trained on 0.04% of the corpus -- on the
        # strength of one section.
        #
        # Dropping columns SILENTLY is what made a gene filter unacceptable
        # before. `process()` therefore reports, per section, how many columns
        # went and what fraction of that section's counts they carried, and the
        # totals land in the manifest.
        self.min_panels_per_gene = int(min_panels_per_gene)
        if self.min_panels_per_gene < 1:
            raise ValueError(
                f"min_panels_per_gene must be >= 1, got "
                f"{self.min_panels_per_gene}."
            )
        # An explicit vocabulary OVERRIDES the threshold. Both merely decide
        # `self.gene_vocab`; filtering each section down to it is one shared
        # code path either way. This is what lets two blobs built over
        # DIFFERENT section sets share one vocabulary -- required, because
        # `gene_ids` are baked into stored sections, so checkpoints trained on
        # blobs with different vocabularies have non-comparable V-wide weights.
        self.gene_vocab_override = (
            None if gene_vocab is None else list(gene_vocab)
        )
        if self.gene_vocab_override is not None and not self.cross_panel:
            raise ValueError(
                "gene_vocab was supplied but cross_panel=False. A shared "
                "vocabulary only has meaning on the cross-panel path, where "
                "`gene_ids` index it."
            )

        # ---- which identity counts as "the batch" ---------------------------
        # "batch"          -> uns['batch'], the legacy behaviour. Correct for
        #                     the paper's setting (1-3 sections, one dataset).
        # "dataset_batch"  -> f"{dataset_id}_{batch}", the composite key.
        #
        # At corpus scale the legacy key is WRONG, not merely coarse:
        # `uns['batch']` is a within-dataset counter, so `batch0` is used by 80
        # different datasets and 627 of 636 sections carry a label that collides
        # across datasets. That hands one decoder-covariate embedding row to 80
        # unrelated experiments and tells the GRL adversary they are the same
        # batch. It also hides a holdout completely: keyed on `uns['batch']`,
        # 219/219 of TERRA's held-out sections look already-seen; keyed on the
        # composite, 0/219 do.
        #
        # TERRA resolved this identically on this same corpus --
        # `batch_id_key = f"{uns['dataset_id']}_{uns['batch']}"`
        # (terra/src/terra/tokenizers/cell_tokenizers.py:1097), which is also
        # the key its split files use.
        #
        # Default stays "batch" so existing blobs, variants and the paper's
        # reproduction are untouched; `process()` WARNS whenever labels actually
        # collide, so the corpus case cannot pass silently again.
        if batch_key not in ("batch", "dataset_batch"):
            raise ValueError(
                f"batch_key must be 'batch' or 'dataset_batch', got "
                f"{batch_key!r}."
            )
        self.batch_key = batch_key

        # force_reload is consulted by OnDiskDataset._process via the base
        # Dataset; store it so `process()` gating matches PyG semantics.
        self.force_reload = overwrite

        # ... except it does NOT survive. `OnDiskDataset.__init__` takes no
        # `force_reload` argument, and the base `Dataset.__init__` assigns
        # `self.force_reload = force_reload` from its own default of False
        # (torch_geometric/data/dataset.py:109) -- AFTER the line above, and
        # before `_process()` reaches `if not self.force_reload and
        # files_exist(...)` (:255). So `overwrite=True` silently did nothing
        # whenever a blob already existed. Every test passed only because each
        # builds into a fresh tmp dir.
        #
        # Benign until now; actively unsafe with cross_panel, because
        # `gene_ids` are baked into each stored section. Rebuilding after
        # changing `exclude_sections` (and therefore the vocabulary) would have
        # silently kept the old rows, leaving every section's ids pointing at
        # the wrong genes -- a wrong answer, not a crash.
        #
        # Remove the gate artifacts instead, which is what `overwrite` promises.
        # `processed_paths` is exactly [store, manifest.json]; the store must go
        # too, not just the manifest, or `process()` would re-run and APPEND to
        # the existing rows. `self.backend` is normally set by the base init, so
        # set it here for `processed_file_names`, which is backend-dependent.
        # `resume` overrides `overwrite`: the whole point is to keep the rows
        # already written. Pass 2 skips the completed prefix; see `process()`.
        self.resume = bool(resume)
        if overwrite and not self.resume:
            self.backend = backend
            for _path in self.processed_paths:
                _p = Path(_path)
                if _p.is_dir():
                    shutil.rmtree(_p)
                elif _p.exists():
                    _p.unlink()

        if overwrite and self.resume:
            self.backend = backend
            print("OnDiskDatasetBlob: resume=True, so overwrite=True is NOT "
                  "clearing the store — completed sections will be kept and "
                  "the build will continue after them.")

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
        name = f"{self.name}{getattr(self, 'output_suffix', '')}"
        return str(gold / "on-disk-PyG-dataset-blob" / name)

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

        # Drop excluded sections BEFORE the vocabulary is computed, so V is the
        # union over what is actually built (see `cross_panel` in __init__).
        if self.exclude_sections or self.include_sections:
            names = self.exclude_sections or self.include_sections
            what = ("exclude_sections" if self.exclude_sections
                    else "include_sections")
            selected = self._resolve_section_selection(raw_files, names, what)
            if self.exclude_sections:
                kept = [f for f in raw_files if f not in selected]
            else:
                kept = [f for f in raw_files if f in selected]
            dropped = [_rel_section_name(f, self.raw_dir)
                       for f in raw_files if f not in kept]
            if not kept:
                raise ValueError(
                    f"{what} leaves no sections to build.")
            print(f"{what}: building {len(kept)} of {len(kept) + len(dropped)} "
                  f"section(s); leaving out {len(dropped)}"
                  + (f": {sorted(dropped)}" if len(dropped) <= 12 else ""))
            raw_files = kept

        # ---------------- Pass 1: gene panel + label vocab (streamed) ------
        self.gene_panel = None
        self.label_categories = {ln: set() for ln in self.label_names}
        # rel path -> sidecar relpath during pass 1; remapped to
        # {resolved adata_batch_id -> relpath} once ids are final, because
        # that is how `inference_data_dict_to_adata` looks it up
        # (type_conversions.py:440).
        obs_index: dict = {}
        # cross_panel only: accumulate the vocabulary and remember each
        # section's own gene order. Strings are held only until pass 1 ends,
        # then collapsed to integer ids (636 sections x ~5k genes x int32 is
        # ~13 MB, versus ~32 MB of interned strings).
        vocab: set = set()
        var_index_per_batch: dict = {}
        # Which sections carry each label, so a partially-present label can be
        # refused at build time rather than at training time. See the check
        # after pass 1.
        _label_present: dict = {}
        _label_absent: dict = {}
        # Derived ids and rel paths in raw_files order, so uniqueness can be
        # checked once pass 1 has seen every section.
        _derived_ids: List[int] = []
        _rels_pass1: List[str] = []

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
            _rel = _rel_section_name(adata_batch_file, self.raw_dir)
            _derived_ids.append(batch_id)
            _rels_pass1.append(_rel)

            # Sidecar named by POSITION, not by `adata_batch_id`. The id is
            # parsed from `uns['batch']`, which is a within-dataset counter:
            # across hst_corpus_110m 561 of 636 sections share an id with
            # another (80 of them are id 0). Naming the file by the id made 80
            # sections overwrite ONE pickle -- and this is where the expert
            # labels live, so NMI/ARI would have scored against another
            # section's annotations. Position is unique by construction.
            obs_rel = f"obs/section_{len(_rels_pass1) - 1:05d}.pkl"
            with open(meta / obs_rel, "wb") as f:
                pickle.dump(adata_batch.obs.copy(), f)
            obs_index[_rel] = obs_rel

            if self.cross_panel:
                # Union, not intersection. Intersection is not merely lossy
                # across the corpus, it is EMPTY: 46 distinct panels whose
                # global intersection is 0 genes (`_panelsurvey_summary.json`).
                # No `.var` metadata comparison either -- panels legitimately
                # disagree there, and the only per-gene fact this path needs is
                # the name, which is what the vocabulary is keyed on.
                names = list(adata_batch.var.index)
                if len(set(names)) != len(names):
                    raise ValueError(
                        f"{Path(adata_batch_file).name} has duplicate gene "
                        f"names in .var; the vocabulary maps name -> column, "
                        f"so duplicates would make `gene_ids` ambiguous."
                    )
                # Keyed by rel path for the same reason as the obs sidecar:
                # keyed by `adata_batch_id` only 75 of 636 entries survived,
                # which silently produced a 5,106-gene vocabulary over 7
                # "panels" instead of 9,574 over 46 -- and left pass 2 slicing
                # each section with another section's column list.
                var_index_per_batch[_rel] = names
                vocab.update(names)
            elif self.gene_panel is None:
                self.gene_panel = adata_batch.var
            else:
                if set(self.gene_panel.index) != set(adata_batch.var.index):
                    raise ValueError(
                        "All batches must share the same gene panel "
                        "(gene SETS differ between batches). Build with "
                        "cross_panel=True to store each section at its own "
                        "width with a `gene_ids` index into a shared "
                        "vocabulary."
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
                    _label_absent.setdefault(label_name, []).append(
                        _rel_section_name(adata_batch_file, self.raw_dir))
                    continue
                _label_present.setdefault(label_name, []).append(
                    _rel_section_name(adata_batch_file, self.raw_dir))
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

        # A label present on SOME sections and absent on others builds happily
        # and then fails at TRAINING time, which is the worst possible place for
        # it: the corpus build is ~48 h and not resumable.
        #
        # The mechanism: `SetExperimentDataKeys.set_node_labels` returns the real
        # one-hot when `y_<label>` exists and a (N, 0) placeholder when it does
        # not, and PyG's `collate` takes its key set from the FIRST `Data` in a
        # block and demands the rest match (torch_geometric/data/collate.py:95).
        # So a block whose first section lacks the label proceeds silently
        # WITHOUT labels, while one whose first section has it raises
        # KeyError on the first section that does not. Which happens depends on
        # the shuffle, so a corpus run would die at an arbitrary point in epoch
        # 1 -- or, worse, train some blocks unlabelled without complaint.
        #
        # hst_corpus_110m is exactly this case: `cell_type` is on 143 of 636
        # sections. Refuse at build time and name the fix.
        for label_name in self.label_names:
            have = _label_present.get(label_name, [])
            lack = _label_absent.get(label_name, [])
            if have and lack:
                raise ValueError(
                    f"Label {label_name!r} is present on {len(have)} section(s) "
                    f"and ABSENT on {len(lack)} (e.g. present {have[:2]}, absent "
                    f"{lack[:2]}). A partially-present label produces a blob "
                    f"that builds cleanly and then fails during training, "
                    f"because a block mixing the two cannot be collated.\n"
                    f"  Either drop it from `label_names` -- expert labels reach "
                    f"the predicted AnnData through the per-section `.obs` "
                    f"sidecars, not through `y_*`, so NMI/ARI benchmarking is "
                    f"unaffected -- or restrict the build to sections that "
                    f"carry it via `exclude_sections`."
                )

        # ---- `adata_batch_id` must be unique per SECTION -------------------
        # It is parsed from `uns['batch']`, a WITHIN-DATASET counter, so on a
        # multi-dataset corpus it collides hard: 561 of hst_corpus_110m's 636
        # sections share an id with another section, 80 of them on id 0.
        #
        # A colliding id is not merely untidy — it is unusable. Per-cell
        # `adata_batch_ids` is what predict uses to look up a cell's `.obs`
        # (`type_conversions.py:440`), so with collisions a cell cannot be
        # traced back to its own section at all.
        #
        # Single-dataset blobs (the paper's setting) are unaffected: their ids
        # are already unique, so the parsed value is kept and variant
        # selections like `test_batch_idx=[3, 1]` keep meaning exactly what
        # they meant. Only when the parsed ids collide are they replaced, by
        # POSITION in the sorted file list — deterministic, and unique by
        # construction. It is announced, never silent.
        self._section_id_by_rel: dict = {}
        _dupe_ids = len(_derived_ids) != len(set(_derived_ids))
        if _dupe_ids:
            from collections import Counter as _C
            _worst = _C(_derived_ids).most_common(3)
            print(
                f"\n*** adata_batch_id collisions: {len(_derived_ids)} sections "
                f"map to only {len(set(_derived_ids))} distinct ids "
                f"(worst: {_worst}).\n"
                f"    uns['batch'] is a within-dataset counter, so it cannot key "
                f"a multi-dataset corpus.\n"
                f"    Reassigning ids by position in the sorted file list so "
                f"every section is addressable.\n",
                flush=True,
            )
            self._section_id_by_rel = {r: i for i, r in enumerate(_rels_pass1)}
        else:
            self._section_id_by_rel = dict(zip(_rels_pass1, _derived_ids))
        if len(set(self._section_id_by_rel.values())) != len(_rels_pass1):
            raise AssertionError(
                "adata_batch_id is still not unique after reassignment; refusing "
                "to build a blob whose cells cannot be traced to their section."
            )

        # Sidecars were written under positional names; re-key the INDEX by the
        # resolved id, which is what predict looks a cell's obs up by.
        obs_index = {
            int(self._section_id_by_rel[r]): rel_path
            for r, rel_path in obs_index.items()
        }

        # ---- finalise the cross-panel vocabulary --------------------------
        if self.cross_panel:
            # NOTE: no local `import pandas as pd` here. A function-level import
            # would make `pd` local to the whole of process(), shadowing the
            # module-level import and making the EARLIER use of `pd` in the
            # pass-1 label loop raise UnboundLocalError.
            #
            # SORTED, so the vocabulary is a deterministic function of the gene
            # names alone -- independent of file order, of which section happens
            # to be read first, and of set iteration order. A build must be
            # reproducible: `gene_ids` are baked into every stored section, so a
            # vocabulary that reshuffled between builds would silently
            # invalidate an existing blob (and any checkpoint trained on it,
            # whose V-wide weight rows are indexed by exactly these ids).
            # An explicit vocabulary wins; otherwise apply the panel threshold.
            # Both paths only decide WHICH names are in `gene_vocab` -- the
            # per-section filtering below is identical either way.
            if self.gene_vocab_override is not None:
                keep_names = set(self.gene_vocab_override)
                absent = sorted(keep_names - vocab)
                self.gene_vocab = pd.Index(
                    sorted(self.gene_vocab_override), name="gene")
                print(f"Cross-panel vocabulary: {len(self.gene_vocab)} genes "
                      f"(supplied, not derived).")
                if absent:
                    # Not an error: a shared vocabulary is deliberately built
                    # over a LARGER section set than this blob, so a holdout
                    # blob is expected to be missing some genes. It is worth
                    # stating, because those V rows can receive no gradient
                    # from this blob.
                    print(f"  {len(absent)} of them appear in NO built section "
                          f"and so will train on nothing here "
                          f"(e.g. {absent[:5]}).")
            elif self.min_panels_per_gene > 1:
                # A PANEL is a distinct gene set, not a section: a gene in 150
                # sections that all share one panel is still a one-panel gene.
                # Derived from `var_index_per_batch`, which pass 1 already
                # holds, so this costs no extra read.
                panels = {frozenset(names) for names in var_index_per_batch.values()}
                panel_count: dict = {}
                for pan in panels:
                    for g in pan:
                        panel_count[g] = panel_count.get(g, 0) + 1
                keep_names = {g for g in vocab
                              if panel_count.get(g, 0) >= self.min_panels_per_gene}
                if not keep_names:
                    raise ValueError(
                        f"min_panels_per_gene={self.min_panels_per_gene} leaves "
                        f"an EMPTY vocabulary over {len(panels)} distinct "
                        f"panels. Lower the threshold."
                    )
                self.gene_vocab = pd.Index(sorted(keep_names), name="gene")
                print(f"Cross-panel vocabulary: {len(self.gene_vocab)} of "
                      f"{len(vocab)} union genes appear in >= "
                      f"{self.min_panels_per_gene} of {len(panels)} distinct "
                      f"panels.")
            else:
                keep_names = None          # keep everything; no filtering
                self.gene_vocab = pd.Index(sorted(vocab), name="gene")

            pos = {g: i for i, g in enumerate(self.gene_vocab)}
            # With a filter, a section keeps only its in-vocabulary columns.
            # `_gene_cols_per_batch` records WHICH native columns survived so
            # pass 2 can slice the stored tensors to match -- `gene_ids` must
            # index the stored columns exactly, which `_stamp_gene_ids` then
            # asserts.
            if keep_names is None:
                self._gene_ids_per_batch = {
                    rel: np.fromiter((pos[g] for g in names), dtype=np.int64,
                                     count=len(names))
                    for rel, names in var_index_per_batch.items()
                }
                self._gene_cols_per_batch = {}
            else:
                self._gene_ids_per_batch = {}
                self._gene_cols_per_batch = {}
                for rel, names in var_index_per_batch.items():
                    cols = [i for i, g in enumerate(names) if g in keep_names]
                    if not cols:
                        raise ValueError(
                            f"Section {rel!r} has no gene in "
                            f"the vocabulary, so it would be stored with zero "
                            f"columns. Lower min_panels_per_gene, or exclude "
                            f"the section explicitly."
                        )
                    self._gene_cols_per_batch[rel] = np.asarray(cols, dtype=np.int64)
                    self._gene_ids_per_batch[rel] = np.fromiter(
                        (pos[names[i]] for i in cols), dtype=np.int64,
                        count=len(cols),
                    )

            widths = sorted({len(v) for v in self._gene_ids_per_batch.values()})
            print(f"  stored per-section widths {widths[0]}..{widths[-1]} "
                  f"({len(widths)} distinct) over "
                  f"{len(self._gene_ids_per_batch)} sections.")
            with open(meta / "gene_vocab.pkl", "wb") as f:
                pickle.dump(self.gene_vocab, f)
            # `gene_panel.pkl` is what predict() reads to recover gene names
            # (run_squint.py, "Resolved gene names ... from gene_panel.pkl").
            # Write the vocabulary in that shape so the existing consumer keeps
            # working; in cross-panel mode it is the vocabulary, not one panel.
            self.gene_panel = pd.DataFrame(index=self.gene_vocab)

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
        section_dataset_ids: List[Optional[str]] = []
        section_rels: List[str] = []
        # Per-section accounting for the vocabulary cap; empty when nothing
        # was dropped. Reported below and recorded in the manifest.
        vocab_drops: List[dict] = []
        # In cross-panel mode `process_anndata_batch` must NOT reindex. Its
        # only use of `self.gene_panel` is to reindex each section onto a single
        # canonical panel (in_memory_dataset_blob.py:381-385) -- exactly what we
        # are replacing -- and with the vocabulary now stored there, leaving it
        # set would reindex every section up to the full width V, reintroducing
        # the padding this design exists to avoid. None disables that branch
        # and touches nothing else in the inherited method.
        _panel_for_pass2 = self.gene_panel
        if self.cross_panel:
            self.gene_panel = None

        # Per-section timing with a running ETA. A corpus build is SERIAL and
        # cannot resume — `process()` appends, so a partial build has to be
        # discarded — which makes a multi-hour run with no progress signal a
        # bad bet: a stall is indistinguishable from slow work until the
        # wall-clock limit kills it. The cost is one `time.time()` per section.
        _t_build0 = time.time()
        _cells_done = 0

        # ---- resume: skip the completed prefix ----------------------------
        # Section i always lands at row i (raw_files is sorted and pass 2 walks
        # it in order), so "already done" is a prefix and the accumulators can
        # be restored wholesale from the progress file rather than recomputed
        # by deserializing rows.
        _fingerprint = self._build_fingerprint(
            [_rel_section_name(f, self.raw_dir) for f in raw_files])
        _n_done, _state = self._load_resume_state(_fingerprint)
        if _n_done:
            section_ids = list(_state["section_ids"])[:_n_done]
            node_counts = list(_state["node_counts"])[:_n_done]
            batch_labels = list(_state["batch_labels"])[:_n_done]
            section_rels = list(_state["section_rels"])[:_n_done]
            section_dataset_ids = list(_state["section_dataset_ids"])[:_n_done]
            vocab_drops = [d for d in _state["vocab_drops"]
                           if d["rel"] in set(section_rels)]
            _cells_done = sum(node_counts)
            print(f"RESUMING: {_n_done} of {len(raw_files)} sections already "
                  f"built ({_cells_done:,} cells); continuing from "
                  f"{_rel_section_name(raw_files[_n_done], self.raw_dir)!r}.",
                  flush=True)

        for _i_sec, adata_batch_file in enumerate(raw_files):
            if _i_sec < _n_done:
                continue
            print(f"Processing {adata_batch_file}...")
            _t_sec = time.time()
            # Inherited; pre_filter/pre_transform already applied inside it
            # (in_memory_dataset_blob.py:579-585).
            data_batch = self.process_anndata_batch(adata_batch_file)
            rel = _rel_section_name(adata_batch_file, self.raw_dir)
            section_rels.append(rel)
            # Apply the id resolved in pass 1. `process_anndata_batch` stamps
            # the value parsed from `uns['batch']`, which is not unique across
            # a multi-dataset corpus; pass 1 has already decided whether to
            # keep it or renumber positionally.
            _resolved = self._section_id_by_rel.get(rel)
            if _resolved is None:
                raise KeyError(
                    f"No section id recorded for {rel!r}. Pass 1 and pass 2 "
                    f"iterate the same file list, so this means they disagreed "
                    f"about which sections exist."
                )
            data_batch.adata_batch_id = int(_resolved)
            if self.cross_panel:
                # Slice BEFORE stamping: `gene_ids` must index the columns that
                # actually get stored, and `_stamp_gene_ids` asserts exactly
                # that, so it doubles as the guard that the two agree.
                drop = self._slice_to_vocab(data_batch, rel)
                if drop is not None:
                    vocab_drops.append(drop)
                self._stamp_gene_ids(data_batch, rel)
            # Append as a serialized row. schema=object => serialize() is a
            # passthrough and the DB pickles the Data (list attrs survive).
            self.append(self.serialize(data_batch))
            section_ids.append(int(data_batch.adata_batch_id))
            section_dataset_ids.append(_section_dataset_id(data_batch))
            # Cheap per-section facts recorded HERE so consumers never have
            # to deserialize a row just to learn them (see `manifest` below).
            node_counts.append(int(data_batch["x_cell_gene_counts"].shape[0]))
            batch_labels.append(_section_batch_identity(data_batch, self.batch_key))
            del data_batch   # never hold two sections at once

            _n = node_counts[-1]
            _cells_done += _n
            _dt = time.time() - _t_sec
            _elapsed = time.time() - _t_build0
            _left = len(raw_files) - (_i_sec + 1)
            # Rate over sections done THIS run, not since row 0 — a resumed run
            # has done fewer than `_i_sec + 1`.
            _this_run = (_i_sec + 1) - _n_done
            _eta = (_elapsed / max(_this_run, 1)) * _left
            print(
                f"  [{_i_sec + 1}/{len(raw_files)}] {rel} — {_n:,} cells in "
                f"{_dt:.1f}s ({_n / max(_dt, 1e-9):,.0f} cells/s); "
                f"elapsed {_elapsed / 60:.1f} min, "
                f"{_cells_done:,} cells done, ETA {_eta / 60:.1f} min",
                flush=True,
            )
            self._save_resume_state(
                _fingerprint, _i_sec + 1, section_ids, node_counts,
                batch_labels, section_rels, section_dataset_ids, vocab_drops,
                obs_index,
            )

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
        if self.cross_panel:
            self.gene_panel = _panel_for_pass2

        _warn_on_batch_label_collisions(
            batch_labels, section_dataset_ids, self.batch_key,
        )
        if vocab_drops:
            _report_vocab_drops(vocab_drops, len(self.gene_vocab))

        manifest = {
            # 3 adds the cross-panel fields; 4 adds `section_rels` (the only
            # corpus-unique section name), the vocabulary policy and its
            # per-section cost, and `batch_key`. `_load_sidecars` reads every
            # field with .get(), so v2/v3 blobs keep loading unchanged.
            "manifest_version": 4,
            "name": self.name,
            "cross_panel": self.cross_panel,
            "gene_vocab_size": (
                int(len(self.gene_vocab)) if self.gene_vocab is not None else None
            ),
            "min_panels_per_gene": self.min_panels_per_gene,
            "gene_vocab_overridden": self.gene_vocab_override is not None,
            "vocab_drops": vocab_drops,
            "batch_key": self.batch_key,
            "excluded_sections": sorted(self.exclude_sections),
            # DB row idx -> "subdir/file.h5ad". The ONLY corpus-unique section
            # name: file stems collide (40 stems over 162 of 636 corpus files,
            # `adata_batch0` alone in 8 datasets) and `uns['batch']` collides
            # harder still (75 values for 636 sections). Everything that has to
            # name a specific row -- `exclude_sections`, and the section-
            # restricted loaders that make whole-section validation affordable
            # -- resolves through this.
            "section_rels": section_rels,
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

        # The manifest is the completion marker; the progress file has no
        # further use and leaving it would invite a "resume" of a finished
        # build.
        _pf = Path(self.processed_dir) / self._PROGRESS_FILE
        if _pf.exists():
            _pf.unlink()

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #
    _PROGRESS_FILE = "_build_progress.json"

    def _build_fingerprint(self, rels: List[str]) -> dict:
        """
        Everything that must be identical for a resumed build to be coherent.

        `gene_ids` are baked into each stored section, so resuming a build whose
        vocabulary differs from the completed rows' would leave earlier sections
        indexing one gene set and later ones another — silently, with no shape
        error anywhere. The vocabulary is therefore hashed, not just counted,
        and the ordered file list is included because pass 2 relies on section i
        always landing at row i.
        """
        import hashlib
        vocab = list(self.gene_vocab) if self.gene_vocab is not None else []
        h = hashlib.sha256("\n".join(vocab).encode()).hexdigest()[:16]
        return {
            "cross_panel": bool(self.cross_panel),
            "min_panels_per_gene": int(self.min_panels_per_gene),
            "batch_key": self.batch_key,
            "gene_vocab_size": len(vocab),
            "gene_vocab_sha": h,
            "feature_names": list(self.feature_names),
            "label_names": list(self.label_names),
            "section_rels": list(rels),
        }

    def _load_resume_state(self, fingerprint: dict):
        """
        How many sections are already written, and their accumulated metadata.

        Returns `(n_done, state)` — `(0, None)` when there is nothing usable.
        A fingerprint mismatch is a hard error rather than a silent restart:
        the caller asked to resume, and quietly rebuilding 636 sections while
        reporting success is the kind of surprise this whole exercise has been
        about.
        """
        pf = Path(self.processed_dir) / self._PROGRESS_FILE
        if not self.resume or not pf.exists():
            return 0, None
        with open(pf) as fh:
            state = json.load(fh)
        old = state.get("fingerprint", {})
        drift = {k: (old.get(k), fingerprint[k]) for k in fingerprint
                 if k != "section_rels" and old.get(k) != fingerprint[k]}
        if old.get("section_rels") != fingerprint["section_rels"]:
            drift["section_rels"] = (
                f"{len(old.get('section_rels') or [])} sections",
                f"{len(fingerprint['section_rels'])} sections",
            )
        if drift:
            raise ValueError(
                f"Cannot resume: the build parameters changed since the "
                f"progress file was written. Differences (was -> now): {drift}. "
                f"Resuming would leave earlier sections indexing a different "
                f"gene vocabulary than later ones, with nothing to raise. "
                f"Delete {pf} and rebuild from scratch."
            )
        n_done = int(state.get("n_done", 0))
        # Trust the store, not the bookkeeping: only count sections the DB
        # actually holds.
        try:
            n_rows = len(self)
        except Exception:  # noqa: BLE001 - store not readable yet
            n_rows = n_done
        if n_rows < n_done:
            print(f"OnDiskDatasetBlob: progress file claims {n_done} sections "
                  f"but the store holds {n_rows}; continuing from {n_rows}.")
            n_done = n_rows
        return n_done, state

    def _save_resume_state(self, fingerprint, n_done, section_ids, node_counts,
                           batch_labels, section_rels, section_dataset_ids,
                           vocab_drops, obs_index) -> None:
        """Written after EVERY section, atomically, so a kill at any point
        leaves a usable record."""
        pf = Path(self.processed_dir) / self._PROGRESS_FILE
        tmp = pf.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump({
                "fingerprint": fingerprint,
                "n_done": n_done,
                "section_ids": section_ids,
                "node_counts": node_counts,
                "batch_labels": batch_labels,
                "section_rels": section_rels,
                "section_dataset_ids": section_dataset_ids,
                "vocab_drops": vocab_drops,
                "obs_index": {str(k): v for k, v in obs_index.items()},
            }, fh)
        tmp.replace(pf)          # atomic on POSIX

    def _resolve_section_selection(self, raw_files, names, what):
        """
        Which of `raw_files` a list of section names refers to.

        Names are REL PATHS ("subdir/file.h5ad"), or bare stems where those are
        unambiguous. Shared by `exclude_sections` and `include_sections` so the
        two cannot drift apart in what they accept.

        Two hard failures, both because a quiet near-miss here mis-scopes an
        entire build: a stem matching several files raises rather than matching
        all of them (40 stems cover 162 of the corpus's 636 files --
        `adata_batch0` appears in 8 datasets), and a name matching nothing
        raises rather than being ignored.
        """
        rel_of = {f: _rel_section_name(f, self.raw_dir) for f in raw_files}
        by_stem: dict = {}
        for f in raw_files:
            by_stem.setdefault(Path(f).stem, []).append(f)

        selected, missing, ambiguous = set(), [], {}
        rels = set(rel_of.values())
        for n in names:
            hit = [f for f in raw_files if rel_of[f] == n]
            if hit:
                selected.update(hit)
                continue
            stem_hits = by_stem.get(n, [])
            if len(stem_hits) == 1:
                selected.update(stem_hits)
            elif len(stem_hits) > 1:
                ambiguous[n] = sorted(rel_of[f] for f in stem_hits)
            else:
                missing.append(n)
        if ambiguous:
            raise ValueError(
                f"{what} names file stems that match more than one section: "
                f"{ambiguous}. Stems are not unique across a corpus. Use the "
                f"relative path ('subdir/file.h5ad') so the selection is exact."
            )
        if missing:
            raise ValueError(
                f"{what} names sections that are not in {self.raw_dir}: "
                f"{sorted(missing)[:10]}. Refusing to build, because a typo "
                f"here would silently change which sections are built (and, "
                f"with cross_panel, the vocabulary derived from them). "
                f"{len(rels)} sections are available."
            )
        return selected

    def _slice_to_vocab(self, data_batch, rel: str) -> Optional[dict]:
        """
        Drop a section's out-of-vocabulary gene columns, cross-panel only.

        No-op (returns None) unless the vocabulary actually filters — with
        `min_panels_per_gene=1` and no override, `_gene_cols_per_batch` is empty
        and a build is bit-identical to before.

        Every gene-width attribute must be sliced together, or `gene_ids` would
        index one tensor correctly and another wrongly. Rather than trusting a
        fixed key list, this finds gene-width tensors by SHAPE — any `x_*` whose
        column count equals the section's native width — and raises on an `x_*`
        that is 2-D with some other width. A silent skip there would store a
        tensor whose columns no longer line up with the ids, which is a wrong
        answer rather than a crash. `y_*` are label-width and untouched.

        Returns the per-section drop accounting, including the fraction of the
        section's total counts carried by the dropped columns — the number that
        makes the loss auditable rather than merely reported.
        """
        cols = self._gene_cols_per_batch.get(rel)
        if cols is None:
            return None
        native = int(data_batch["x_cell_gene_counts"].shape[1])
        if len(cols) == native:
            return None

        keep = torch.as_tensor(cols, dtype=torch.long)
        gene_keys = [k for k in list(data_batch.keys()) if k.startswith("x_")]
        total = None
        dropped_mass = None
        for k in gene_keys:
            t = data_batch[k]
            if not torch.is_tensor(t) or t.dim() != 2:
                continue
            if t.shape[1] != native:
                raise ValueError(
                    f"{rel}: gene-width attribute {k!r} has {t.shape[1]} "
                    f"columns but the section's native width is {native}. "
                    f"Every gene-width tensor must share one column order for "
                    f"`gene_ids` to index them all."
                )
            if k == "x_cell_gene_counts":
                total = float(t.sum())
                dropped_mass = total - float(t.index_select(1, keep).sum())
            data_batch[k] = t.index_select(1, keep)

        frac = 0.0 if not total else dropped_mass / total
        return {
            "rel": rel,
            "native_cols": native,
            "kept_cols": int(len(cols)),
            "dropped_cols": int(native - len(cols)),
            "dropped_frac_counts": round(frac, 6),
        }

    def _stamp_gene_ids(self, data_batch, rel: str) -> None:
        """
        Attach this section's vocabulary ids, cross-panel builds only.

        Shape is `[1, G_sec]`, NOT `[G_sec]`, and that is load-bearing rather
        than stylistic. PyG decides "is this a node attribute?" purely by
        testing `size(cat_dim) == num_nodes` -- a SHAPE test, with no reference
        to the key. A flat `[G_sec]` therefore becomes a node attribute by
        accident whenever a section has as many cells as genes, after which
        collation and `NeighborLoader` slice it like per-cell data and the ids
        silently stop corresponding to the columns of `x`. At corpus scale
        (thousands of genes, thousands of cells) that coincidence is ordinary,
        not exotic. A leading singleton dim can only collide when a section
        holds exactly one cell. Verified both ways in
        `tests/test_gene_vocab_pyg_contract.py`.

        The name is also checked there: `Data.__inc__` adds an offset to any key
        CONTAINING "batch", which is what silently shifted `adata_batch_ids`
        0/1/2 into 0/2/5. "gene_ids" is inert under that rule.
        """
        ids = self._gene_ids_per_batch.get(rel)
        if ids is None:
            raise KeyError(
                f"No gene ids recorded for section {rel!r}. Pass 1 builds the "
                f"map from the same file list as pass 2, so this means the two "
                f"passes disagreed about which sections exist."
            )
        n_genes = int(data_batch["x_cell_gene_counts"].shape[1])
        if len(ids) != n_genes:
            raise ValueError(
                f"Section {rel!r}: pass 1 recorded {len(ids)} genes but the "
                f"processed counts have {n_genes} columns. `gene_ids` must "
                f"index the columns of `x_cell_gene_counts` exactly, or the "
                f"per-block union would scatter counts into the wrong genes."
            )
        data_batch.gene_ids = torch.as_tensor(ids, dtype=torch.long).unsqueeze(0)

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
        _manifest: dict = {}
        if mpath.exists():
            with open(mpath) as f:
                _manifest = json.load(f)
            self.section_ids = _manifest.get("section_ids", [])
            self.node_counts = _manifest.get("node_counts", None)
            self.batch_labels = _manifest.get("batch_labels", None)
            # manifest_version >= 4. `section_rels` is the corpus-unique row
            # name that `section_rows_for()` resolves against; None on older
            # blobs, which then cannot address rows by name.
            self.section_rels = _manifest.get("section_rels", None)
            self.batch_key = _manifest.get("batch_key", "batch")
            self.min_panels_per_gene = _manifest.get("min_panels_per_gene", 1)
            # Blobs built before the file container existed have no `container`
            # field and are sqlite by construction.
            self.container = _manifest.get("container", "sqlite")

        gp = meta / "gene_panel.pkl"
        if gp.exists():
            with open(gp, "rb") as f:
                self.gene_panel = pickle.load(f)

        # Cross-panel fields. All read with .get()/exists() so a v2 blob (and
        # every blob built before this) reopens exactly as it did.
        self.cross_panel = bool(_manifest.get("cross_panel", False))
        self.gene_vocab = None
        gv = meta / "gene_vocab.pkl"
        if gv.exists():
            with open(gv, "rb") as f:
                self.gene_vocab = pickle.load(f)

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

    def batch_label_to_dense(self, rows: Optional[List[int]] = None) -> dict:
        """
        {batch label -> dense id} over `rows`, or over every section when
        `rows` is None. Ids are assigned over the SORTED unique labels.

        Sorted-unique matches `build_batch_one_hot_from_obs`
        (initializers/initialize.py), so a section gets the same dense id
        whether it is loaded through the streaming or the in-memory path.

        `rows` exists for the zero-shot setting, and it is the difference
        between a correct evaluation and a silently wrong one. Restricted to
        the TRAIN sections, the map has one row per batch the model actually
        sees, so every embedding row receives gradient and the mean-embedding
        fallback for novel batches is uncontaminated. A held-out section's label
        is then absent, `_stamp_batch_ids` marks it unseen, and the model
        substitutes that mean instead of an arbitrary reference batch — which is
        what "this section was never in training" is supposed to mean.

        Derived over the whole blob instead, a held-out label would resolve as
        KNOWN and index a row that never trained; and because ids come from
        sorted-unique, adding held-out labels also RENUMBERS the shared ones, so
        every cell's covariate silently moves. That exact failure has already
        occurred here once (fixed in 13f437b).
        """
        labels_all = self.get_batch_labels()
        if rows is None:
            labels = sorted(set(labels_all))
        else:
            labels = sorted({str(labels_all[int(r)]) for r in rows})
        return {lbl: i for i, lbl in enumerate(labels)}

    def section_rows_for(self, names) -> List[int]:
        """
        DB row indices for the named sections.

        `names` are rel paths ("subdir/file.h5ad"), or bare file stems where
        those are unambiguous. Raises on a name that matches nothing or matches
        several rows — this is what a train/val/test split is built from, so a
        silent partial match would mis-scope an entire evaluation.

        Requires manifest_version >= 4 (`section_rels`); older blobs have no
        corpus-unique row name to resolve against.
        """
        rels = getattr(self, "section_rels", None)
        if not rels:
            raise RuntimeError(
                "This blob's manifest has no 'section_rels' (built before "
                "manifest_version 4), so its rows cannot be addressed by name. "
                "Rebuild the blob to use section-restricted loaders."
            )
        by_rel = {r: i for i, r in enumerate(rels)}
        by_stem: dict = {}
        for i, r in enumerate(rels):
            by_stem.setdefault(Path(r).stem, []).append(i)

        rows, missing, ambiguous = [], [], {}
        for n in names:
            if n in by_rel:
                rows.append(by_rel[n])
                continue
            hits = by_stem.get(n, [])
            if len(hits) == 1:
                rows.append(hits[0])
            elif len(hits) > 1:
                ambiguous[n] = [rels[i] for i in hits]
            else:
                missing.append(n)
        if missing:
            raise KeyError(
                f"{len(missing)} section name(s) not in this blob, e.g. "
                f"{sorted(missing)[:5]}."
            )
        if ambiguous:
            raise ValueError(
                f"Ambiguous section stems (use the rel path): {ambiguous}."
            )
        return sorted(set(rows))

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


GENE_WIDTH_ATTRS = ("x", "x_cell_gene_counts")


def scatter_sections_to_columns(
        views,
        section_gene_ids,
        target_ids: torch.Tensor,
        gene_width_attrs=GENE_WIDTH_ATTRS,
    ):
    """
    Scatter each section's gene-width tensors onto a shared column set, in place.

    Shared by the two cross-panel consumers, which differ only in what they
    scatter ONTO:

      * `KSectionBlockLoader._widen_to_block_union` passes the union of the
        block's panels, keeping a block as narrow as its contents allow;
      * `initialize_databatch` passes `arange(V)` -- the whole vocabulary --
        because predict collates every section into one object and the model's
        weights are V-wide, so no gather is needed there at all.

    `target_ids` MUST be sorted ascending: `searchsorted` locates each section's
    columns in it, which is what avoids a dict and a Python loop over genes.

    Returns `(panel_masks, panel_of_section)`:
      panel_masks      BoolTensor[P, W] one row per DISTINCT panel among `views`
      panel_of_section list[int]        index into `views` -> panel row

    Masks are per PANEL, not per cell: [P, W] is kilobytes where a per-cell
    [N, W] mask would be ~530 MB at K=8, and the per-cell view is one gather at
    mini-batch size. Deduplicating matters because panels repeat heavily -- 150
    corpus sections share the 4,949-gene panel.
    """
    W = int(target_ids.numel())
    masks: List[torch.Tensor] = []
    panel_of_section: List[int] = []
    key_to_panel: dict = {}

    for view, ids in zip(views, section_gene_ids):
        cols = torch.searchsorted(target_ids, ids)
        g = int(ids.numel())
        if int(cols.max()) >= W or not bool((target_ids[cols] == ids).all()):
            raise ValueError(
                "A section carries gene ids absent from the target column set. "
                "The vocabulary is the union over the sections in the build, so "
                "this cannot happen unless the blob and the vocabulary sidecar "
                "disagree -- rebuild with overwrite=True."
            )

        for key in [k for k in gene_width_attrs if k in view]:
            src = view[key]
            if src.shape[1] != g:
                raise ValueError(
                    f"`{key}` has {src.shape[1]} columns but the section "
                    f"records {g} gene ids. They must correspond exactly, or "
                    f"counts would scatter into the wrong genes."
                )
            # Unmeasured positions stay EXACTLY zero. Load-bearing:
            # `read_depth = batch.x.sum(dim=-1)` (vqniche_dual.py:476) is only
            # the measured-gene depth because of it, so no separate masked sum
            # is needed there.
            wide = src.new_zeros((src.shape[0], W))
            wide[:, cols] = src
            view[key] = wide

        # Exactly ONE of the two is ever present, so this widens one matrix per
        # section rather than two. `SetExperimentDataKeys.forward` deletes every
        # `x_*` key at the end (dataset/transforms.py:874-877), so once it has
        # run the raw counts are gone and only `x` remains. With NO section
        # transform they are never copied into `x` and they ARE the features.
        # Hence the allow-list rather than a hardcoded "x".

        key = tuple(ids.tolist())
        panel = key_to_panel.get(key)
        if panel is None:
            panel = len(masks)
            key_to_panel[key] = panel
            m = torch.zeros(W, dtype=torch.bool)
            m[cols] = True
            masks.append(m)
        panel_of_section.append(panel)

    return torch.stack(masks), panel_of_section


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
        section_rows: Optional[List[int]] = None,
        max_cells_per_block: Optional[int] = None,
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
        # Cap a block by CELLS as well as by section count, because section
        # count does not bound memory on a real corpus.
        #
        # A block widens every section to the UNION of its panels, so its
        # gene-width tensor is (sum of cells) x (union width) x 4 bytes. Across
        # hst_corpus_110m sections span 1,049 to 1,617,614 cells and stored
        # widths 169 to 9,540, so a fixed K has a ~20x spread between a typical
        # block and the worst one: measured over random shuffles, K=8 gives a
        # median block of 26.7 GB and a worst case of 144 GB. A 96 GB run died
        # at 102 GB (TERM_MEMLIMIT) for exactly this reason.
        #
        # Budgeting cells bounds it: union width can never exceed the
        # vocabulary, so cells x V x 4 is a hard ceiling per block, and the
        # budget can be set from available memory. It also MIXES BETTER than a
        # small K -- the median section is 89,752 cells, so a budget sized for
        # one large section admits several typical ones, and section mixing is
        # what supplies the batch-integration signal.
        self.max_cells_per_block = (int(max_cells_per_block)
                                    if max_cells_per_block else None)
        self.batch_size = int(batch_size)
        self.num_neighbors = list(num_neighbors)
        self.shuffle = shuffle
        self.seed = seed
        self.input_mask_attr = input_mask_attr
        # Restrict the epoch to these DB rows. None => the whole blob.
        #
        # Without this a val pass builds EVERY block in the corpus to reach the
        # val cells -- ~51 min at 636 sections, which is what forced validation
        # off entirely. With whole-section splits (TERRA's design, and the
        # SQUINT paper's for query-to-reference) the val loader visits only the
        # val sections, so a val epoch costs seconds and honest validation
        # curves become affordable at corpus scale.
        if section_rows is not None:
            n = len(dataset)
            bad = [r for r in section_rows if not (0 <= int(r) < n)]
            if bad:
                raise IndexError(
                    f"section_rows out of range for a {n}-section blob: "
                    f"{bad[:5]}."
                )
            section_rows = sorted({int(r) for r in section_rows})
            if not section_rows:
                raise ValueError(
                    "section_rows is empty — a loader over no sections would "
                    "yield an empty epoch silently."
                )
        self.section_rows = section_rows
        # Fail fast on transforms that cannot be applied per section.
        _reject_global_scope_transforms(section_transform, where="section_transform")
        self.section_transform = section_transform
        # Corpus-wide {batch label -> dense id}. MUST span every section, not
        # just this block's — see `_stamp_batch_ids`.
        self.batch_label_to_dense = batch_label_to_dense
        self.unknown_batch_label_dense_id = int(unknown_batch_label_dense_id)
        self.prefetch = bool(prefetch)
        # Resolved once from the DATASET, which is where the build-time flag
        # lives -- the loader has no say in whether a blob is cross-panel.
        self.cross_panel = bool(getattr(dataset, "cross_panel", False))
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
        order = (list(range(len(self.dataset))) if self.section_rows is None
                 else list(self.section_rows))
        if self.shuffle:
            random.Random(self.seed + self._epoch).shuffle(order)
        K = self.sections_per_block
        if not self.max_cells_per_block:
            return [order[i:i + K] for i in range(0, len(order), K)]

        # Both limits apply: at most K sections, and at most
        # `max_cells_per_block` cells. Cell counts come from the manifest, so
        # this costs no I/O. A single section over budget still forms its own
        # block — a section is the indivisible unit here.
        counts = self.dataset.get_node_counts()
        blocks: List[List[int]] = []
        cur: List[int] = []
        cur_cells = 0
        for r in order:
            n = int(counts[r])
            if cur and (cur_cells + n > self.max_cells_per_block
                        or len(cur) >= K):
                blocks.append(cur)
                cur, cur_cells = [], 0
            cur.append(r)
            cur_cells += n
        if cur:
            blocks.append(cur)
        return blocks

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

    # -- cross-panel: per-block gene union ---------------------------------- #

    # Gene-width attributes, by NAME. Deliberately an explicit allow-list rather
    # than "any [n, G] tensor": `y_cell_types` is [n, n_classes], so a shape
    # heuristic would silently widen the labels of any dataset whose class count
    # happened to equal its gene count.
    _GENE_WIDTH_ATTRS = GENE_WIDTH_ATTRS

    def _widen_to_block_union(self, views, section_gene_ids):
        """
        Scatter each section onto the union of the block's panels, in place.

        Returns `(union, panel_masks, panel_of_section)`:
          union            LongTensor[W]    sorted global vocabulary ids
          panel_masks      BoolTensor[P, W] one row per DISTINCT panel
          panel_of_section list[int]        section index -> panel row

        Masks are per PANEL, not per cell. A per-cell [N, W] mask is ~530 MB at
        K=8, while [P, W] is kilobytes and the per-cell gather is trivial at
        mini-batch size. Deduplicating matters because panels repeat heavily --
        150 corpus sections share the 4,949-gene panel, so a block of 8 such
        sections has P=1.

        Block width is max-driven, not mean-driven: mixing a 169-gene section
        with a 4,949-gene one gives W ~= 4,949 and leaves the narrow section's
        rows ~97% zeros. That waste is deliberate -- grouping same-panel
        sections into a block would minimise W but remove exactly the
        cross-panel mixing the batch-correction terms need. Panel-aware block
        grouping is a possible future knob.
        """
        # `unique` returns SORTED values, which is what makes `searchsorted`
        # inside `scatter_sections_to_columns` valid -- no dict and no Python
        # loop over genes.
        union = torch.unique(torch.cat(section_gene_ids))
        return (union,) + scatter_sections_to_columns(
            views, section_gene_ids, union, self._GENE_WIDTH_ATTRS,
        )

    def _stamp_panel_attrs(self, block: Data, union, panel_masks,
                           panel_of_section, sizes: List[int]) -> None:
        """
        Stamp the cross-panel attributes onto an already-collated block.

        AFTER collation for the same reason as `_stamp_batch_ids`: none of these
        are per-section quantities, and `panel_id` is per-CELL only once the
        sections are concatenated.

        `gene_ids` is [1, W], not [W]. PyG decides "is this a node attribute?"
        by testing `size(cat_dim) == num_nodes` -- a shape test that never looks
        at the key -- so a flat [W] becomes per-cell data whenever W equals the
        block's cell count, after which `NeighborLoader` slices it and the ids
        stop corresponding to the columns of `x`. At corpus scale (thousands of
        genes, thousands of cells per block) that is an ordinary coincidence.
        Asserted both ways in `tests/test_gene_vocab_pyg_contract.py`.
        `panel_masks[P, W]` needs no such guard: it would require P == N_blk.
        """
        block.gene_ids = union.reshape(1, -1)
        block.panel_masks = panel_masks
        block.panel_id = torch.cat([
            torch.full((n,), int(p), dtype=torch.long)
            for p, n in zip(panel_of_section, sizes)
        ])

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

        # `gene_ids` must be read off the SECTION, before `_block_view`. Step 5b
        # of that method drops every tensor whose first dim != num_nodes, and
        # `gene_ids` is [1, G] by design (see `_stamp_gene_ids`), so it is
        # already gone by the time the view is returned. Convenient rather than
        # awkward: it means the per-section ids can never reach
        # `Batch.from_data_list`, where sections of differing width would either
        # fail to collate or concatenate into nonsense. The block gets its own.
        views = []
        section_gene_ids: List = []
        for r in block_rows:
            section = self._fetch_section(r)
            if self.cross_panel:
                ids = getattr(section, "gene_ids", None)
                if ids is None:
                    raise KeyError(
                        f"Section (row {r}) has no `gene_ids`, but this blob is "
                        f"cross_panel. It was probably built before "
                        f"cross_panel=True; rebuild with overwrite=True."
                    )
                section_gene_ids.append(ids.reshape(-1))
            views.append(self._block_view(section, r))
            del section                    # the view shares its tensors already

        sizes = [int(v.num_nodes) for v in views]

        # Widen each section onto the block's gene union BEFORE collation --
        # `Batch.from_data_list` cannot concatenate `x` of differing widths,
        # which is exactly what native-width storage produces.
        union = panel_masks = panel_of_section = None
        if self.cross_panel:
            union, panel_masks, panel_of_section = self._widen_to_block_union(
                views, section_gene_ids,
            )

        block = Batch.from_data_list(views)              # edges offset per section
        # Must come AFTER collation — see `_stamp_batch_ids` for why.
        self._stamp_batch_ids(block, block_rows, sizes)
        if self.cross_panel:
            self._stamp_panel_attrs(block, union, panel_masks,
                                    panel_of_section, sizes)

        # Hand NeighborLoader a plain `Data`, NOT the `Batch` that from_data_list
        # returned. This is load-bearing, not tidying.
        #
        # `Batch.batch_size` is a PROPERTY returning num_graphs. `Data` has no such
        # property. NeighborLoader sets `batch_size` on its output to the seed count,
        # but when the input graph is a Batch that assignment is shadowed by the
        # property, so the mini-batch reports K (sections per block) instead of the
        # real seed count. Measured on PyG 2.6.1: a 2-section block yields
        # mb.batch_size == 2 while mb.input_id.numel() == 256.
        #
        # That value is not cosmetic. `VQNiche_Dual._step` (:570) reads
        # `batch_size = batch.batch_size` and it slices the losses:
        # `batch_pred_attr_and_target_attr` computes
        # `pred_attr = batch_xhat[:batch_size]` / `target_attr = batch_x[:batch_size]`
        # (the MAIN cell-branch NB loss), and :711-712 / :740-742 slice the adversary
        # input and the quantizer terms the same way. Left uncorrected, streaming
        # trains on the first K cells of every mini-batch instead of all 256 — it runs,
        # the loss descends, and nothing raises.
        #
        # The in-memory path already avoids this by rebuilding a plain Data before the
        # loader (in_memory_datamodule.py:420, `Data(**data_dict_for_loader)`); this
        # mirrors it. `ptr` is dropped because it has length K+1 and is not a node
        # attribute, which would confuse PyG's node/edge inference; `batch` is
        # per-node and harmless to keep.
        block = Data(**{k: v for k, v in block.to_dict().items() if k != "ptr"})
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

        MEASURED CAVEAT (R0, 39 sections / 519,058 cells, K=8): this does NOT
        currently hide the boundaries. GPU utilisation was 8.1% with 53% of
        samples at exactly 0%, and the 179 idle stretches came to 4.97 per
        epoch — precisely the 5 blocks/epoch. The budget is ~1.5 min/epoch of
        consuming against ~3.1 min/epoch of block-build work, so ONE producer
        thread is structurally ~2x too slow and queue depth alone cannot fix
        it. Note also that the GIL claim in the paragraph above is an
        assertion, not a measurement: if the per-section transform is
        Python/numpy CPU work rather than I/O, threads cannot help here at all.

        Deliberately left as-is for now. `REMAINING_WORK.md` item 0d records
        the proposed fix (profile the build first, then producer parallelism or
        a worker process), and why it is postponed: this is tested code where a
        threading bug would silently yield a wrong block rather than raise, and
        per item 0b a corpus epoch already fits in one overnight job, so the
        upside is ~2x on something that is not blocking.
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
            try:
                yield item
            finally:
                # RELEASE BEFORE THE NEXT `q.get()`.
                #
                # `item` is a local of this generator's frame and survives the
                # yield, so it kept the block alive even after `__iter__` had
                # done `del nl; del block; gc.collect()` -- the collect could
                # not free a block this frame still referenced. That is one
                # resident block MORE than the design's three (consumer, queue,
                # producer-building), and 4 x ~19 GB matches the 75.4 GB peak
                # measured with prefetch on against 34.8 GB without it.
                #
                # `finally` rather than a plain assignment so it also runs when
                # the consumer ABANDONS this generator, which Lightning does on
                # every `max_steps` stop: without it the in-flight block leaked
                # for the lifetime of the iterator.
                item = None

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
            try:
                for mini_batch in nl:
                    yield mini_batch
            finally:
                # Release the block BEFORE the next one is built, and do it in
                # a `finally` so it also runs when the consumer ABANDONS this
                # generator -- which Lightning does at every `max_steps` stop.
                #
                # `del block` alone did not release anything. `nl` holds the
                # block as its dataset and was only rebound on the next
                # iteration, and PyG's NeighborLoader/NeighborSampler keep it in
                # reference cycles that only the cyclic collector breaks, so
                # freeing was deferred to whenever CPython happened to run a
                # generational collection.
                #
                # Measured by `BaseModel._tensor_census` on the corpus: 22.18
                # GiB in 260 live tensors at step 200, 35.54 GiB in 392 at step
                # 1,200, with four-plus DISTINCT blocks resident in a run where
                # exactly one should be. All grad=False, graph=False -- not a
                # retained autograd graph, just blocks never freed. Fixing it
                # took the single-threaded arm from 13.1 MB/step to -1.3, i.e.
                # flat, and its peak from 63.3 GB to 34.8.
                #
                # The collect is per BLOCK, not per step: blocks are seconds
                # apart and the heap holds hundreds of objects, so it is cheap.
                del nl
                del block
                gc.collect()
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

        # Only the rows this loader will actually visit. Everything else keeps a
        # 0 that is never read: `__len__` and `__iter__` both index `counts`
        # through `_block_row_lists()`, which yields only these rows. Streaming
        # the untouched rows too would reintroduce exactly the corpus-wide pass
        # `section_rows` exists to avoid.
        rows = (list(range(len(self.dataset))) if self.section_rows is None
                else list(self.section_rows))
        print(
            f"KSectionBlockLoader: counting '{self.input_mask_attr}' seeds by "
            f"streaming {len(rows)} of {len(self.dataset)} sections once (split "
            f"masks are produced at load time, so they are not in the "
            f"manifest). Cached for the rest of this loader's life."
        )
        counts: List[int] = [0] * len(self.dataset)
        for row in rows:
            view = self._block_view(self._fetch_section(row), row)
            if self.input_mask_attr in view:
                counts[row] = int(view[self.input_mask_attr].sum())
            else:
                # Transform produced no such mask -> treat every cell as a
                # seed, matching NeighborLoader's behaviour for input_nodes=None.
                counts[row] = int(view.num_nodes)
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
        max_cells_per_block: Optional[int] = None,
        split_sections: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        # {split -> [section name, ...]} for whole-section splits; see
        # `_resolve_section_rows`. None keeps the in-section cell-mask splits.
        self.split_sections = dict(split_sections or {}) or None
        self._section_rows_cache: dict = {}
        self.edge_index_name = edge_index_name
        self.sections_per_block = int(sections_per_block)
        self.max_cells_per_block = max_cells_per_block
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
        # With whole-section splits the map is derived from the TRAIN sections
        # ONLY, and that is deliberate rather than an optimisation. Deriving it
        # over the whole blob would size the decoder-covariate embedding to
        # every section, leaving one untrained row per held-out section (219 of
        # 635 on TERRA's split — 34% of the embedding, all of it random init).
        # Those rows would then pollute the mean-embedding fallback AND make
        # held-out labels resolve as "seen", so a zero-shot evaluation would
        # quietly condition on noise. Restricted to train, every row trains and
        # held-out sections take the unseen path by construction.
        if batch_label_to_dense is not None:
            self.batch_label_to_dense = batch_label_to_dense
        elif self.split_sections:
            self.batch_label_to_dense = dataset.batch_label_to_dense(
                rows=self._resolve_section_rows("train"),
            )
        else:
            self.batch_label_to_dense = dataset.batch_label_to_dense()
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
        rows = self._resolve_section_rows(split)
        if self.split_sections and rows is None and mask_attr is not None:
            # Whole-section splits are configured but this split is not among
            # them, so it would silently fall back to an in-section cell mask
            # spanning EVERY section -- drawing this split's cells from the
            # TRAINING sections. The run then looks healthy (the metrics appear
            # and improve) while the number it reports is leaked. Refuse.
            raise ValueError(
                f"split_sections is configured ({sorted(self.split_sections)}) "
                f"but names no sections for {split!r}, which would fall back to "
                f"an in-section '{mask_attr}' over all {len(self.dataset)} "
                f"sections and draw {split} cells from the training sections. "
                f"Name {split!r}'s sections explicitly, or drop split_sections "
                f"entirely to use cell-mask splits throughout."
            )
        if rows is not None and mask_attr is not None:
            # Whole-section splits and in-section cell masks are two different
            # answers to "which cells are val". Applying both would intersect
            # them and silently validate on a fraction of the named sections,
            # so a named section set takes over the split entirely.
            mask_attr = None
        return KSectionBlockLoader(
            section_rows=rows,
            dataset=self.dataset,
            edge_index_name=self.edge_index_name,
            sections_per_block=self.sections_per_block,
            max_cells_per_block=self.max_cells_per_block,
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

    def _resolve_section_rows(self, split: str) -> Optional[List[int]]:
        """
        DB rows for a split's sections, or None for "the whole blob".

        `split_sections` maps split name -> section names (rel paths, or
        unambiguous stems), resolved once against the manifest. Train defaults
        to "everything not named by another split" so a caller only has to list
        the held-out sections — listing train's 417 by hand invites a typo that
        would silently shrink the training set.
        """
        spec = self.split_sections
        if not spec:
            return None
        cache = self._section_rows_cache
        if split in cache:
            return cache[split]

        named = {s: self.dataset.section_rows_for(v)
                 for s, v in spec.items() if v}
        if split == "train" and "train" not in named:
            claimed = {r for s, rows in named.items() if s != "train"
                       for r in rows}
            rows = [r for r in range(len(self.dataset)) if r not in claimed]
            if not rows:
                raise ValueError(
                    "split_sections leaves no sections for training."
                )
        else:
            rows = named.get(split)
        cache[split] = rows
        if rows is not None:
            print(f"OnDiskStreamingDataModule: split {split!r} -> "
                  f"{len(rows)} of {len(self.dataset)} sections.")
        return rows

    def train_dataloader(self):
        return self._make_loader(split="train")

    def val_dataloader(self):
        return self._make_loader(split="val")

    def test_dataloader(self):
        return self._make_loader(split="test")

    def predict_dataloader(self):
        return self._make_loader(split="predict")
