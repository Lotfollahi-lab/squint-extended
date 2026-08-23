"""
Per-block gene-union assembly for cross-panel blobs.

`OnDiskDatasetBlob(cross_panel=True)` stores each section at its own gene width,
so `Batch.from_data_list` can no longer concatenate the views -- their `x` have
different column counts. `_build_block` therefore scatters each section onto the
union of its block's panels first, and stamps three attributes:

    gene_ids    [1, W]     global vocabulary id of each block column
    panel_masks [P, W]     bool, one row per DISTINCT panel in the block
    panel_id    [N_blk]    which panel each cell belongs to

The single most important test here is `test_single_panel_blocks_are_identical`:
with one panel the whole path must reduce to the identity, so any numerical
drift is a bug in the new code rather than a property of the data. That turns
"did I break the tested path?" into an assertion.
"""
import numpy as np
import pytest
import torch

from vqniche.dataset.on_disk_dataset import KSectionBlockLoader, OnDiskDatasetBlob
from vqniche.dataset.transforms import SetExperimentDataKeys

GRAPH_KWARGS = {
    "coord_type": "generic",
    "spatial_key": "spatial",
    "n_neighs_list": [4],
    "radius_list": None,
    "include_self_loop": True,
    "k": {},
}
EDGE = "spatial_n_neighs_4"


def _write(root, panels, name="xp", seed=0, fill=None):
    """
    `panels`: section index -> gene names. `fill`: section index -> constant
    value written into every cell of that section, so a mis-scatter is visible.
    """
    import anndata as ad

    rng = np.random.default_rng(seed)
    silver = root / "silver" / name
    silver.mkdir(parents=True)
    for i, genes in panels.items():
        n = 12 + i
        if fill is not None:
            counts = np.full((n, len(genes)), float(fill[i]), dtype="float32")
        else:
            counts = rng.poisson(0.8, size=(n, len(genes))).astype("float32") + 1.0
        a = ad.AnnData(counts)
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


def _transform():
    """The section transform the real training path uses to produce `data.x`."""
    return SetExperimentDataKeys(
        feature_names=["X"], label_name="cell_types", edge_index_name=EDGE,
    )


def _loader(blob, K, transform=None):
    return KSectionBlockLoader(
        dataset=blob,
        edge_index_name=EDGE,
        sections_per_block=K,
        batch_size=4,
        num_neighbors=[2],
        shuffle=False,
        section_transform=transform,
        batch_label_to_dense=blob.batch_label_to_dense(),
    )


def _block(blob, K, transform=None):
    """The first assembled block, before NeighborLoader."""
    ld = _loader(blob, K, transform)
    return next(iter(ld._blocks()))


# Mirrors xhc38-4b_1p: 319 a strict subset of 419.
NESTED = {0: ["a", "b", "c", "d"], 1: ["a", "b"]}
# Mirrors xhb42-3b_1p: exclusive genes both ways.
DISJOINT = {0: ["a", "b", "shared"], 1: ["shared", "y", "z"]}


# --------------------------------------------------------------------------- #
# the equivalence guard
# --------------------------------------------------------------------------- #

def test_single_panel_blocks_are_identical(tmp_path):
    """
    One panel => W == G, an all-ones mask and an identity permutation, so the
    cross-panel path must be numerically indistinguishable from the old one.
    """
    same = {0: ["g0", "g1", "g2"], 1: ["g0", "g1", "g2"], 2: ["g0", "g1", "g2"]}
    plain = _build(tmp_path / "a", _write(tmp_path / "a", same, name="p"))
    xp = _build(tmp_path / "b", _write(tmp_path / "b", same, name="q"),
                cross_panel=True)

    b_plain = _block(plain, K=3, transform=_transform())
    b_xp = _block(xp, K=3, transform=_transform())

    torch.testing.assert_close(b_plain.x, b_xp.x)
    torch.testing.assert_close(b_plain.edge_index, b_xp.edge_index)
    torch.testing.assert_close(b_plain.adata_batch_ids, b_xp.adata_batch_ids)
    assert b_xp.gene_ids.tolist() == [[0, 1, 2]]
    assert b_xp.panel_masks.shape == (1, 3) and b_xp.panel_masks.all()
    assert b_xp.panel_id.tolist() == [0] * b_xp.num_nodes


# --------------------------------------------------------------------------- #
# union geometry
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("panels,expect_W", [(NESTED, 4), (DISJOINT, 5)])
def test_block_width_is_the_union(tmp_path, panels, expect_W):
    blob = _build(tmp_path, _write(tmp_path, panels), cross_panel=True)
    blk = _block(blob, K=2, transform=_transform())
    assert blk.x.shape[1] == expect_W
    assert blk.gene_ids.shape == (1, expect_W)


def test_nested_union_equals_the_wider_panel_disjoint_exceeds_it(tmp_path):
    """The distinction between the two real datasets: xhc38 vs xhb42."""
    nested = _build(tmp_path / "n", _write(tmp_path / "n", NESTED, name="n"),
                    cross_panel=True)
    disj = _build(tmp_path / "d", _write(tmp_path / "d", DISJOINT, name="d"),
                  cross_panel=True)
    bn = _block(nested, K=2, transform=_transform())
    bd = _block(disj, K=2, transform=_transform())
    assert bn.x.shape[1] == max(len(v) for v in NESTED.values())      # 4 == 4
    assert bd.x.shape[1] > max(len(v) for v in DISJOINT.values())     # 4 > 3


def test_gene_ids_are_sorted(tmp_path):
    """`searchsorted` in `_widen_to_block_union` is only valid if they are."""
    blob = _build(tmp_path, _write(tmp_path, DISJOINT), cross_panel=True)
    ids = _block(blob, K=2, transform=_transform()).gene_ids[0]
    assert torch.equal(ids, torch.sort(ids).values)
    assert ids.unique().numel() == ids.numel()


# --------------------------------------------------------------------------- #
# scatter correctness
# --------------------------------------------------------------------------- #

def test_counts_land_in_the_right_columns(tmp_path):
    """
    Each section is filled with its own constant, so a mis-scatter cannot pass:
    a cell must hold its own fill value in its measured columns and zero
    elsewhere.
    """
    panels = {0: ["a", "b", "shared"], 1: ["shared", "y", "z"]}
    fills = {0: 3.0, 1: 7.0}
    name = _write(tmp_path, panels, fill=fills)
    blob = _build(tmp_path, name, cross_panel=True)
    blk = _block(blob, K=2, transform=_transform())

    vocab = list(blob.gene_vocab)
    col = {g: vocab.index(g) for g in vocab}
    for sec_pos, genes in panels.items():
        rows = (blk.adata_batch_ids_raw == sec_pos) if hasattr(
            blk, "adata_batch_ids_raw") else (blk.batch == sec_pos)
        sub = blk.x[rows]
        assert sub.numel() > 0
        for g in vocab:
            observed = sub[:, col[g]]
            if g in genes:
                assert torch.all(observed == fills[sec_pos]), g
            else:
                assert torch.all(observed == 0.0), g


def test_unmeasured_positions_are_exactly_zero(tmp_path):
    """
    `read_depth = batch.x.sum(dim=-1)` (vqniche_dual.py:476) is the
    measured-gene depth ONLY because unmeasured columns are exact zeros.
    """
    panels = {0: ["a", "b", "shared"], 1: ["shared", "y", "z"]}
    name = _write(tmp_path, panels, fill={0: 2.0, 1: 5.0})
    blob = _build(tmp_path, name, cross_panel=True)
    blk = _block(blob, K=2, transform=_transform())
    per_cell = blk.panel_masks[blk.panel_id]
    assert torch.all(blk.x[~per_cell] == 0.0)
    assert torch.all(blk.x[per_cell] != 0.0)
    # read_depth from the widened x equals the per-section measured total
    depth = blk.x.sum(dim=-1)
    assert torch.allclose(depth[blk.panel_id == blk.panel_id[0]].unique(),
                          depth[blk.panel_id == blk.panel_id[0]][:1])


# --------------------------------------------------------------------------- #
# panel masks
# --------------------------------------------------------------------------- #

def test_panel_mask_marks_exactly_the_measured_genes(tmp_path):
    panels = {0: ["a", "b", "shared"], 1: ["shared", "y", "z"]}
    blob = _build(tmp_path, _write(tmp_path, panels), cross_panel=True)
    blk = _block(blob, K=2, transform=_transform())
    vocab = list(blob.gene_vocab)
    for sec_pos, genes in panels.items():
        panel = int(blk.panel_id[blk.batch == sec_pos][0])
        mask = blk.panel_masks[panel]
        named = {vocab[i] for i in mask.nonzero().flatten().tolist()}
        assert named == set(genes)


def test_identical_panels_deduplicate(tmp_path):
    """Three sections, two distinct panels -> P == 2, not 3."""
    panels = {0: ["a", "b"], 1: ["a", "b"], 2: ["a", "c"]}
    blob = _build(tmp_path, _write(tmp_path, panels), cross_panel=True)
    blk = _block(blob, K=3, transform=_transform())
    assert blk.panel_masks.shape[0] == 2
    assert int(blk.panel_id[blk.batch == 0][0]) == int(blk.panel_id[blk.batch == 1][0])
    assert int(blk.panel_id[blk.batch == 2][0]) != int(blk.panel_id[blk.batch == 0][0])


def test_panel_id_is_per_cell_and_matches_its_section(tmp_path):
    panels = {0: ["a", "b"], 1: ["b", "c"]}
    blob = _build(tmp_path, _write(tmp_path, panels), cross_panel=True)
    blk = _block(blob, K=2, transform=_transform())
    assert blk.panel_id.numel() == blk.num_nodes
    for sec_pos in (0, 1):
        vals = blk.panel_id[blk.batch == sec_pos].unique()
        assert vals.numel() == 1, "a section's cells must share one panel"


# --------------------------------------------------------------------------- #
# memory: the raw counts copy
# --------------------------------------------------------------------------- #

def test_only_x_is_present_when_a_transform_ran(tmp_path):
    """
    With a section transform, `x` is the single gene-width attribute a block
    carries -- so `_widen_to_block_union` widens one matrix per section, not two.
    """
    blob = _build(tmp_path, _write(tmp_path, DISJOINT), cross_panel=True)
    blk = _block(blob, K=2, transform=_transform())
    assert "x" in blk
    assert "x_cell_gene_counts" not in blk


def test_raw_counts_kept_and_widened_without_a_transform(tmp_path):
    """
    With no section transform the raw counts ARE the features, so they must be
    widened and kept -- dropping them would empty the block.
    """
    blob = _build(tmp_path, _write(tmp_path, DISJOINT), cross_panel=True)
    blk = _block(blob, K=2, transform=None)
    assert "x_cell_gene_counts" in blk
    assert "x" not in blk
    assert blk["x_cell_gene_counts"].shape[1] == 5      # union of DISJOINT


def test_the_transform_not_cross_panel_drops_the_raw_counts(tmp_path):
    """
    Documents where the drop actually happens, because I got this wrong first
    time and asserted the opposite. `SetExperimentDataKeys.forward` deletes
    EVERY `x_*` key at the end (dataset/transforms.py:874-877), so the raw
    counts never reached a block on either path -- there was no pre-existing
    memory duplication for cross-panel to fix. Both paths must agree.
    """
    same = {0: ["g0", "g1"], 1: ["g0", "g1"]}
    plain = _build(tmp_path / "a", _write(tmp_path / "a", same, name="p"))
    xp = _build(tmp_path / "b", _write(tmp_path / "b", same, name="q"),
                cross_panel=True)
    for blob in (plain, xp):
        blk = _block(blob, K=2, transform=_transform())
        assert "x" in blk
        assert "x_cell_gene_counts" not in blk


# --------------------------------------------------------------------------- #
# NeighborLoader round trip, on real blocks
# --------------------------------------------------------------------------- #

def test_round_trip_preserves_ids_and_masks_slices_panel_id(tmp_path):
    panels = {0: ["a", "b", "shared"], 1: ["shared", "y", "z"]}
    blob = _build(tmp_path, _write(tmp_path, panels), cross_panel=True)
    ld = _loader(blob, K=2, transform=_transform())
    blk = _block(blob, K=2, transform=_transform())
    W = blk.x.shape[1]

    seen = 0
    for mb in ld:
        seen += 1
        torch.testing.assert_close(mb.gene_ids, blk.gene_ids)
        assert mb.panel_masks.shape == (blk.panel_masks.shape[0], W)
        torch.testing.assert_close(mb.panel_masks, blk.panel_masks)
        assert mb.panel_id.numel() == mb.num_nodes
        per_cell = mb.panel_masks[mb.panel_id]
        assert per_cell.shape == mb.x.shape
        assert torch.all(mb.x[~per_cell] == 0.0)
    assert seen > 0


def test_missing_gene_ids_raises_a_clear_error(tmp_path):
    """A blob built before cross_panel, reopened as cross_panel, must not
    silently produce garbage."""
    same = {0: ["g0", "g1"], 1: ["g0", "g1"]}
    blob = _build(tmp_path, _write(tmp_path, same))     # no gene_ids stored
    blob.cross_panel = True                             # simulate a stale blob
    with pytest.raises(KeyError, match="no `gene_ids`"):
        _block(blob, K=2, transform=_transform())
