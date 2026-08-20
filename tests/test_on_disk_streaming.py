"""
Tests for the on-disk streaming path (`vqniche.dataset.on_disk_dataset`).

Each test builds a tiny synthetic corpus in `tmp_path` — a handful of sections,
a few hundred cells, ~20 genes, one spatial graph, no spectral embeddings — so
the whole module runs in seconds and needs no farm data.

Regression coverage, one test per bug fixed:
  - `__len__` must equal the number of mini-batches actually yielded, for
    every K. It previously returned `ceil(total_seeds / batch_size)`, which
    ignores that each block starts a fresh NeighborLoader and so contributes
    its own partial final batch; Lightning takes the value as
    `num_training_batches` and truncates the epoch to it.
  - `__len__` must not read the DB. It previously deserialized every section
    to read one integer each.
  - mini-batches must carry per-cell `adata_batch_ids`, and those ids must NOT
    be shifted by PyG's `__inc__` "batch"-substring rule.
  - transforms that must decide globally (`SubsetHVG`) must be rejected when
    passed as a per-section transform.
  - blobs written before manifest_version 2 must still load.
"""

import json
import math
import pickle

import anndata as ad
import numpy as np
import pytest
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

from vqniche.dataset.on_disk_dataset import (
    KSectionBlockLoader,
    OnDiskDatasetBlob,
    _reject_global_scope_transforms,
    _section_batch_label,
)

# Small enough to stay fast; uneven so partial final batches actually occur.
SECTION_SIZES = (37, 23, 41)
N_GENES = 12
BATCH_SIZE = 8
GRAPH_KWARGS = {
    "coord_type": "generic",
    "spatial_key": "spatial",
    "n_neighs_list": [4],
    "radius_list": None,
    "include_self_loop": True,
    "k": {},          # no spectral embeddings — they are not consumed
}


def _write_silver(root, sizes=SECTION_SIZES, name="testds", seed=0):
    """Write `len(sizes)` synthetic sections and return the corpus root."""
    rng = np.random.default_rng(seed)
    silver = root / "silver" / name
    silver.mkdir(parents=True)
    for i, n in enumerate(sizes):
        counts = rng.poisson(0.6, size=(n, N_GENES)).astype("float32")
        adata = ad.AnnData(counts)
        adata.var.index = [f"gene{j}" for j in range(N_GENES)]
        adata.obs["cell_type"] = rng.choice(["A", "B", "C"], size=n)
        adata.obs["cell_id"] = [f"b{i}_c{j}" for j in range(n)]
        adata.obsm["spatial"] = rng.random((n, 2)) * 100
        adata.uns["batch"] = f"batch{i}"
        adata.uns["dataset_id"] = name
        adata.uns["tissue"] = "synthetic"
        adata.uns["species"] = "synthetic"
        adata.write_h5ad(silver / f"section_{i}.h5ad")
    return name


def _build(root, name):
    return OnDiskDatasetBlob(
        name=name,
        feature_names=["cell_gene_counts"],
        label_names=["cell_types=cell_type"],
        graph_kwargs=GRAPH_KWARGS,
        data_directory_path=root,
        pre_filter=None,
        overwrite=True,
        software_paths={"deepwalk": "", "gosh": ""},
    )


@pytest.fixture(scope="module")
def blob(tmp_path_factory):
    """One built blob shared by the read-only tests."""
    root = tmp_path_factory.mktemp("corpus")
    name = _write_silver(root)
    return _build(root, name)


def _loader(blob, K, **kw):
    kw.setdefault("batch_label_to_dense", blob.batch_label_to_dense())
    return KSectionBlockLoader(
        dataset=blob,
        edge_index_name="spatial_n_neighs_4",
        sections_per_block=K,
        batch_size=BATCH_SIZE,
        num_neighbors=[2],
        shuffle=False,
        **kw,
    )


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def test_manifest_records_per_section_facts(blob):
    """node_counts / batch_labels must be written so nothing has to fetch rows."""
    manifest = json.loads((blob._meta_dir / "manifest.json").read_text())
    assert manifest["manifest_version"] >= 2
    assert manifest["node_counts"] == list(SECTION_SIZES)
    assert manifest["batch_labels"] == [f"batch{i}" for i in range(len(SECTION_SIZES))]
    assert blob.get_node_counts() == list(SECTION_SIZES)


def test_batch_label_to_dense_is_sorted_unique(blob):
    """Dense ids follow sorted-unique order, matching build_batch_one_hot_from_obs."""
    assert blob.batch_label_to_dense() == {"batch0": 0, "batch1": 1, "batch2": 2}


def test_section_batch_label_reads_uns_batch(blob):
    assert [_section_batch_label(blob.get(i)) for i in range(len(blob))] == [
        "batch0", "batch1", "batch2",
    ]


def test_pre_v2_manifest_still_loads(tmp_path):
    """A blob written before manifest_version 2 must degrade, not break."""
    root = tmp_path / "old"
    root.mkdir()
    name = _write_silver(root)
    blob = _build(root, name)

    mpath = blob._meta_dir / "manifest.json"
    manifest = json.loads(mpath.read_text())
    del manifest["node_counts"], manifest["batch_labels"], manifest["manifest_version"]
    mpath.write_text(json.dumps(manifest))

    reopened = OnDiskDatasetBlob(
        name=name,
        feature_names=["cell_gene_counts"],
        label_names=["cell_types=cell_type"],
        graph_kwargs=GRAPH_KWARGS,
        data_directory_path=root,
        pre_filter=None,
        overwrite=False,
        software_paths={"deepwalk": "", "gosh": ""},
    )
    assert reopened.node_counts is None          # absent from the manifest
    assert reopened.get_node_counts() == list(SECTION_SIZES)   # derived instead
    assert reopened.get_batch_labels() == [f"batch{i}" for i in range(3)]


# --------------------------------------------------------------------------- #
# __len__  (the epoch-truncation bug)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("K", [1, 2, 3])
def test_len_equals_batches_actually_yielded(blob, K):
    """
    The regression test for silent epoch truncation.

    Lightning stops the epoch at len(dataloader) (fit_loop.py:253), so any
    under-count drops the tail of every epoch.
    """
    assert len(_loader(blob, K)) == sum(1 for _ in _loader(blob, K))


@pytest.mark.parametrize("K", [1, 2])
def test_len_beats_the_old_global_formula(blob, K):
    """
    Guard against a regression to `ceil(total_seeds / batch_size)`.

    With uneven sections and K < n_sections that formula is strictly smaller,
    so this pins the difference rather than just the value.
    """
    old = math.ceil(sum(SECTION_SIZES) / BATCH_SIZE)
    assert len(_loader(blob, K)) > old


def test_len_reads_nothing_from_the_db(blob, monkeypatch):
    """
    Sizing an epoch used to deserialize the whole corpus, one full section per
    integer read. With counts in the manifest it must touch the DB zero times.
    """
    calls = []
    original = type(blob).get

    def counting_get(self, idx):
        calls.append(idx)
        return original(self, idx)

    monkeypatch.setattr(type(blob), "get", counting_get)
    length = len(_loader(blob, 2))
    assert length > 0
    assert calls == [], f"__len__ fetched {len(calls)} sections"


def test_len_with_a_split_mask_is_still_exact(blob):
    """
    Split masks come from section_transform at load time, so they cannot be in
    the manifest. The count must still match what the loader yields.
    """
    split = T.RandomNodeSplit(num_val=0.2, num_test=0.2)
    kw = {"section_transform": split, "input_mask_attr": "train_mask"}
    assert len(_loader(blob, 2, **kw)) == sum(1 for _ in _loader(blob, 2, **kw))


# --------------------------------------------------------------------------- #
# per-cell batch ids
# --------------------------------------------------------------------------- #

def test_minibatches_carry_per_cell_batch_ids(blob):
    """
    The decoder covariate and adversarial head index an embedding with these;
    the model raises without them (vqniche_dual.py:492-497, :659-666).
    """
    for mb in _loader(blob, 2):
        assert hasattr(mb, "adata_batch_ids")
        assert mb.adata_batch_ids.numel() == mb.num_nodes
        assert mb.adata_batch_ids.dtype == torch.long
        assert mb.adata_batch_ids_unseen_mask.numel() == mb.num_nodes


def test_batch_ids_are_not_shifted_by_pyg_inc(blob):
    """
    PyG's Data.__inc__ offsets any attribute whose key CONTAINS "batch" by
    int(value.max())+1 per list element. Stamping `adata_batch_ids` per section
    and then collating turned ids 0/1/2 into 0/2/5, so stamping happens after
    collation. Only the three real dense ids may ever appear.
    """
    valid = set(blob.batch_label_to_dense().values())
    seen = set()
    for mb in _loader(blob, 3):
        seen |= set(mb.adata_batch_ids.tolist())
    assert seen <= valid, f"ids outside {sorted(valid)} appeared: {sorted(seen)}"
    assert seen == valid


def test_inc_rule_still_applies_to_this_attribute_name():
    """
    Pin the upstream behaviour the fix works around, so a PyG change that makes
    the workaround unnecessary (or breaks it) is noticed here.
    """
    d = Data(x=torch.rand(4, 2), adata_batch_ids=torch.zeros(4, dtype=torch.long))
    assert d.__inc__("adata_batch_ids", d.adata_batch_ids) != 0
    assert d.__inc__("section_row", torch.zeros(4, dtype=torch.long)) == 0


def test_blocks_mix_sections(blob):
    """
    The reason K > 1 exists: a mini-batch must be able to contain cells from
    more than one section, or the batch-correction losses get no signal.
    """
    assert any(len(set(mb.adata_batch_ids.tolist())) > 1 for mb in _loader(blob, 3))


def test_unknown_labels_are_flagged_not_silently_remapped(blob):
    """Predict-time novel sections get the fallback id AND the unseen flag."""
    partial = {"batch0": 0}
    flagged = any(
        mb.adata_batch_ids_unseen_mask.any().item()
        for mb in _loader(blob, 3, batch_label_to_dense=partial)
    )
    assert flagged


def test_list_attrs_are_stripped(blob):
    """NeighborLoader rejects python-list node attrs, so they must not survive."""
    for mb in _loader(blob, 2):
        assert "cell_id" not in mb
        assert "obs_batch" not in mb


# --------------------------------------------------------------------------- #
# transform scope guard
# --------------------------------------------------------------------------- #

def test_subset_hvg_is_rejected_per_section():
    """
    Per-section HVG selects a different gene set per section; widths still
    match so nothing would raise downstream while column j stops meaning the
    same gene. Must be refused up front.
    """
    from vqniche.dataset.transforms import SubsetHVG

    with pytest.raises(ValueError, match="single decision for the whole corpus"):
        _reject_global_scope_transforms(SubsetHVG(n_genes=5))

    with pytest.raises(ValueError, match="SubsetHVG"):
        _reject_global_scope_transforms(
            T.Compose([T.Compose([SubsetHVG(n_genes=5)]), T.NormalizeFeatures()])
        )


def test_per_section_transforms_are_accepted():
    """Split transforms are within-section by design and must stay allowed."""
    _reject_global_scope_transforms(None)
    _reject_global_scope_transforms(T.RandomNodeSplit(num_val=0.1, num_test=0.1))
    _reject_global_scope_transforms(
        T.Compose([T.RandomNodeSplit(num_val=0.1, num_test=0.1)])
    )


def test_loader_rejects_global_transform_at_construction(blob):
    from vqniche.dataset.transforms import SubsetHVG

    with pytest.raises(ValueError, match="cannot be applied per section"):
        _loader(blob, 2, section_transform=SubsetHVG(n_genes=5))


# --------------------------------------------------------------------------- #
# prefetching
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("K", [1, 2])
def test_prefetch_does_not_change_results(blob, K):
    """
    Prefetching is a scheduling change only, so the SEED stream must be
    identical.

    Compared on seed nodes, not `num_nodes`: a mini-batch also contains
    randomly sampled neighbours, so total size differs between any two
    iterations regardless of prefetching. NeighborLoader puts the seeds first,
    so `mb.adata_batch_ids[:mb.batch_size]` is the deterministic part given a
    fixed block partition and shuffle=False.
    """
    def seed_fingerprint(prefetch):
        out = []
        for mb in _loader(blob, K, prefetch=prefetch):
            n_seeds = int(mb.batch_size)
            out.append((n_seeds, mb.adata_batch_ids[:n_seeds].tolist()))
        return out

    assert seed_fingerprint(True) == seed_fingerprint(False)


def test_producer_errors_reach_the_consumer(blob, monkeypatch):
    """A failure on the prefetch thread must not be swallowed."""
    def boom(self, row):
        raise RuntimeError("synthetic producer failure")

    # Patch the fetch funnel, not dataset.get: on a prefetch thread the read
    # goes through a thread-local handle rather than dataset.get.
    monkeypatch.setattr(KSectionBlockLoader, "_fetch_section", boom)
    with pytest.raises(RuntimeError, match="synthetic producer failure"):
        for _ in _loader(blob, 2, prefetch=True):
            pass


def test_prefetch_reads_on_its_own_sqlite_handle(blob):
    """
    Regression test for a bug this suite caught: sqlite3 connections are bound
    to the creating thread, so a prefetch thread reusing the shared handle
    raised ProgrammingError. Each non-main thread must open its own handle.
    """
    n = sum(1 for _ in _loader(blob, 2, prefetch=True))
    assert n == len(_loader(blob, 2, prefetch=True))
    # And the main thread's handle must still work afterwards.
    assert blob.get(0) is not None


# --------------------------------------------------------------------------- #
# obs sidecars
# --------------------------------------------------------------------------- #

def test_obs_is_spilled_per_section_and_lazily_loaded(blob):
    """
    Holding every section's obs would be tens of GB at corpus scale, so it is
    written per section and faulted in on access.
    """
    assert len(list(blob._meta_dir.glob("obs/section_*.pkl"))) == len(SECTION_SIZES)
    obs_map = blob.obs_per_batch_id
    for i, n in enumerate(SECTION_SIZES):
        assert len(obs_map[i]) == n
