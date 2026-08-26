"""
Cross-panel vocabulary build (`cross_panel=True`) on `OnDiskDatasetBlob`.

Covers the build half of union+mask: sections stored at their OWN gene width
with a `gene_ids` index into a corpus-wide vocabulary, replacing the
single-panel hard raise. The per-block union assembly, masked softmax and
masked NB are separate.

Two properties matter more than the rest and are tested first:

  * the default path is BYTE-UNCHANGED. `cross_panel` defaults to False, so
    every existing blob, variant and test keeps hard-raising on mismatched
    panels. A regression here would silently change the paper's data path.
  * the vocabulary is a deterministic function of the gene NAMES. `gene_ids`
    are baked into every stored section, so a vocabulary that reshuffled
    between builds would invalidate an existing blob and any checkpoint trained
    on it, whose V-wide weight rows are indexed by exactly these ids.
"""
import json
import pickle

import anndata as ad
import numpy as np
import pytest
import torch

from vqniche.dataset.on_disk_dataset import OnDiskDatasetBlob

GRAPH_KWARGS = {
    "coord_type": "generic",
    "spatial_key": "spatial",
    "n_neighs_list": [4],
    "radius_list": None,
    "include_self_loop": True,
    "k": {},
}


def _write(root, panels, name="xpanel", seed=0):
    """
    `panels` maps section index -> list of gene names, so each section can carry
    a different panel. Returns the dataset name.
    """
    rng = np.random.default_rng(seed)
    silver = root / "silver" / name
    silver.mkdir(parents=True)
    for i, genes in panels.items():
        n = 20 + i
        a = ad.AnnData(rng.poisson(0.6, size=(n, len(genes))).astype("float32"))
        a.var.index = list(genes)
        a.obs["cell_type"] = rng.choice(["A", "B"], size=n)
        a.obs["cell_id"] = [f"b{i}_c{j}" for j in range(n)]
        a.obsm["spatial"] = rng.random((n, 2)) * 100
        a.uns.update(batch=f"batch{i}", dataset_id=name, tissue="syn", species="syn")
        a.write_h5ad(silver / f"section_{i}.h5ad")
    return name


def _build(root, name, **kw):
    return OnDiskDatasetBlob(
        name=name,
        feature_names=["cell_gene_counts"],
        label_names=["cell_types=cell_type"],
        graph_kwargs=GRAPH_KWARGS,
        data_directory_path=root,
        pre_filter=None,
        overwrite=True,
        software_paths={"deepwalk": "", "gosh": ""},
        **kw,
    )


# Nested, mirroring the real xhc38-4b_1p (319 genes a strict subset of 419).
NESTED = {0: [f"g{i}" for i in range(8)], 1: [f"g{i}" for i in range(5)]}
# True two-way union, mirroring xhb42-3b_1p (88 and 55 exclusive genes).
DISJOINT = {0: ["a", "b", "c", "shared1", "shared2"],
            1: ["shared1", "shared2", "x", "y"]}


# --------------------------------------------------------------------------- #
# the default path must not move
# --------------------------------------------------------------------------- #

def test_mismatched_panels_still_raise_by_default(tmp_path):
    name = _write(tmp_path, DISJOINT)
    with pytest.raises(ValueError, match="must share the same gene panel"):
        _build(tmp_path, name)


def test_single_panel_build_is_unaffected_by_the_new_flag(tmp_path):
    """cross_panel=True on a single-panel corpus must agree with the old path."""
    same = {0: ["g0", "g1", "g2"], 1: ["g0", "g1", "g2"]}
    n1 = _write(tmp_path / "a", same, name="one")
    n2 = _write(tmp_path / "b", same, name="two")
    plain = _build(tmp_path / "a", n1)
    xp = _build(tmp_path / "b", n2, cross_panel=True)
    assert list(plain.gene_panel.index) == list(xp.gene_vocab)
    for row in range(len(plain)):
        torch.testing.assert_close(
            plain[row]["x_cell_gene_counts"], xp[row]["x_cell_gene_counts"]
        )


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("panels,expected", [(NESTED, 8), (DISJOINT, 7)])
def test_vocabulary_is_the_union(tmp_path, panels, expected):
    name = _write(tmp_path, panels)
    blob = _build(tmp_path, name, cross_panel=True)
    assert len(blob.gene_vocab) == expected
    assert set(blob.gene_vocab) == set().union(*[set(v) for v in panels.values()])


def test_vocabulary_is_sorted_and_order_independent(tmp_path):
    """
    Same genes, opposite section order -> identical vocabulary. Guards the
    reproducibility that stored `gene_ids` depend on.
    """
    fwd = {0: ["b", "a"], 1: ["c", "b"]}
    rev = {0: ["c", "b"], 1: ["b", "a"]}
    b1 = _build(tmp_path / "f", _write(tmp_path / "f", fwd, name="f"), cross_panel=True)
    b2 = _build(tmp_path / "r", _write(tmp_path / "r", rev, name="r"), cross_panel=True)
    assert list(b1.gene_vocab) == list(b2.gene_vocab) == ["a", "b", "c"]


def test_sections_keep_their_native_width(tmp_path):
    """
    The whole point of the design: nothing is padded to V. Padding the corpus to
    the 9,571-gene union would take it from ~1.21 TB to ~4.21 TB.
    """
    name = _write(tmp_path, NESTED)
    blob = _build(tmp_path, name, cross_panel=True)
    widths = sorted(int(blob[r]["x_cell_gene_counts"].shape[1]) for r in range(len(blob)))
    assert widths == [5, 8]
    assert len(blob.gene_vocab) == 8


def test_gene_ids_map_columns_to_vocabulary_positions(tmp_path):
    """`gene_ids[0][j]` must be the vocabulary position of column j."""
    name = _write(tmp_path, DISJOINT)
    blob = _build(tmp_path, name, cross_panel=True)
    vocab = list(blob.gene_vocab)
    for row in range(len(blob)):
        sec = blob[row]
        bid = int(sec.adata_batch_id)
        ids = sec.gene_ids
        assert ids.shape == (1, sec["x_cell_gene_counts"].shape[1])
        names = [vocab[i] for i in ids[0].tolist()]
        assert names == list(DISJOINT[bid])


def test_gene_ids_use_the_1xG_layout(tmp_path):
    """
    Shape, not style: a flat [G] becomes a node attribute whenever G equals the
    section's cell count, after which PyG slices it like per-cell data. See
    tests/test_gene_vocab_pyg_contract.py.
    """
    name = _write(tmp_path, NESTED)
    blob = _build(tmp_path, name, cross_panel=True)
    for row in range(len(blob)):
        assert blob[row].gene_ids.ndim == 2
        assert blob[row].gene_ids.shape[0] == 1


def test_duplicate_gene_names_are_refused(tmp_path):
    name = _write(tmp_path, {0: ["g0", "g1", "g1"], 1: ["g0", "g1", "g2"]})
    with pytest.raises(ValueError, match="duplicate gene"):
        _build(tmp_path, name, cross_panel=True)


# --------------------------------------------------------------------------- #
# exclude_sections: the knob that sets V
# --------------------------------------------------------------------------- #

def test_exclude_sections_narrows_the_vocabulary(tmp_path):
    """
    The real case in miniature: one section carries genes nothing else measures
    (like chp60-1b_1p's ~9,366), so excluding it shrinks V.
    """
    panels = {0: ["a", "b"], 1: ["a", "b"], 2: ["a", "b", "wide1", "wide2"]}
    name = _write(tmp_path, panels)
    full = _build(tmp_path, name, cross_panel=True)
    assert len(full.gene_vocab) == 4
    narrow = _build(tmp_path, name, cross_panel=True,
                    exclude_sections=["section_2"])
    assert list(narrow.gene_vocab) == ["a", "b"]
    assert len(narrow) == 2


def test_excluded_section_is_absent_not_silently_truncated(tmp_path):
    """
    Excluding must drop the SECTION, never keep it with its extra genes
    dropped -- that would discard part of a built section's data with no error.
    """
    panels = {0: ["a", "b"], 1: ["a", "b", "wide"]}
    name = _write(tmp_path, panels)
    blob = _build(tmp_path, name, cross_panel=True, exclude_sections=["section_1"])
    assert len(blob) == 1
    assert int(blob[0].adata_batch_id) == 0


def test_typo_in_exclude_sections_raises(tmp_path):
    name = _write(tmp_path, NESTED)
    with pytest.raises(ValueError, match="not in"):
        _build(tmp_path, name, cross_panel=True, exclude_sections=["sectoin_0"])


# --------------------------------------------------------------------------- #
# sidecars / manifest
# --------------------------------------------------------------------------- #

def test_manifest_and_vocab_sidecar(tmp_path):
    name = _write(tmp_path, DISJOINT)
    blob = _build(tmp_path, name, cross_panel=True)
    m = json.load(open(f"{blob.processed_dir}/manifest.json"))
    # v4 added `section_rels`, the vocabulary policy and `batch_key`. The
    # cross-panel fields it inherits from v3 must not have moved.
    assert m["manifest_version"] == 4
    assert m["cross_panel"] is True
    assert m["gene_vocab_size"] == 7
    assert m["min_panels_per_gene"] == 1 and m["vocab_drops"] == []
    assert m["batch_key"] == "batch"
    assert len(m["section_rels"]) == len(blob)
    with open(f"{blob.processed_dir}/gene_vocab.pkl", "rb") as f:
        assert list(pickle.load(f)) == list(blob.gene_vocab)


def test_reopen_restores_the_vocabulary(tmp_path):
    name = _write(tmp_path, DISJOINT)
    built = _build(tmp_path, name, cross_panel=True)
    vocab = list(built.gene_vocab)
    del built
    reopened = OnDiskDatasetBlob(name=name, data_directory_path=tmp_path)
    assert reopened.cross_panel is True
    assert list(reopened.gene_vocab) == vocab


def test_v2_blob_reopens_without_cross_panel_fields(tmp_path):
    """A blob built the old way must reopen with cross_panel False, not crash."""
    name = _write(tmp_path, {0: ["g0", "g1"], 1: ["g0", "g1"]})
    built = _build(tmp_path, name)
    del built
    reopened = OnDiskDatasetBlob(name=name, data_directory_path=tmp_path)
    assert reopened.cross_panel is False
    assert reopened.gene_vocab is None


def test_overwrite_actually_rebuilds(tmp_path):
    """
    Regression: `overwrite=True` silently did nothing when a blob already
    existed. `OnDiskDataset.__init__` takes no `force_reload`, and the base
    `Dataset.__init__` resets `self.force_reload` to False
    (torch_geometric/data/dataset.py:109) after our assignment, so `process()`
    was skipped. Every test passed because each built into a fresh tmp dir.

    Benign before cross_panel; unsafe with it, since `gene_ids` are stored per
    section. Rebuilding with a different vocabulary would have kept the old
    rows and pointed every section's ids at the wrong genes.
    """
    panels = {0: ["a", "b"], 1: ["a", "b", "wide"]}
    name = _write(tmp_path, panels)
    first = _build(tmp_path, name, cross_panel=True)
    assert len(first.gene_vocab) == 3 and len(first) == 2
    del first
    # Same destination, narrower build. Must re-process, not reuse.
    second = _build(tmp_path, name, cross_panel=True, exclude_sections=["section_1"])
    assert list(second.gene_vocab) == ["a", "b"]
    assert len(second) == 1, "process() did not re-run: stale rows were reused"
    # and no duplicated rows from appending onto an existing store
    assert int(second[0].adata_batch_id) == 0
