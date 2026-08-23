"""
PyG contract for the cross-panel (union + mask) block attributes.

This is a GATE, not a feature test: it pins down how PyG classifies the three
new attributes that union+mask needs a block to carry, BEFORE anything is built
on top of them. Two prior bugs in this codebase came from exactly this
inference being wrong-by-assumption:

  * `Data.__inc__` shifts any attribute whose key CONTAINS "batch", which
    silently turned per-section `adata_batch_ids` 0/1/2 into 0/2/5.
  * `Batch.batch_size` is a property returning `num_graphs` and shadows the
    attribute NeighborLoader sets, so the model read K instead of the seed
    count and trained on the first K rows of every mini-batch.

Neither raised. Both were found by measuring. So the attributes are measured
here first.

The three attributes, and why each has the shape it has:

  `gene_ids`    LongTensor[W]      global vocabulary id of each block column
  `panel_masks` BoolTensor[P, W]   measured-gene mask, one row per panel in the
                                   block -- NOT per cell. A per-cell [N, W]
                                   mask would be ~530 MB at K=8; [P, W] is
                                   kilobytes and the per-cell gather is trivial
                                   at mini-batch size.
  `panel_id`    LongTensor[N]      which panel each cell belongs to; the only
                                   one of the three that IS a node attribute.

The hazard this file exists to catch: PyG decides "is this a node attribute?"
by testing `size(cat_dim) == num_nodes`. That is a SHAPE test, not a name test,
so `gene_ids[W]` becomes a node attribute by accident whenever W == N, and
`panel_masks[P, W]` whenever P == N. A block of 5,000 genes and 5,000 cells is
not a contrived scenario at corpus scale. If that happens, NeighborLoader
slices them like node data and the mask silently stops corresponding to the
genes.
"""
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import NeighborLoader


def _section(n_nodes, n_genes, seed=0):
    """One synthetic section, shaped like a real block view."""
    rng = np.random.default_rng(seed)
    src = torch.arange(n_nodes).repeat_interleave(2)
    dst = torch.tensor(rng.integers(0, n_nodes, size=2 * n_nodes), dtype=torch.long)
    d = Data(
        x=torch.tensor(rng.poisson(0.6, size=(n_nodes, n_genes)), dtype=torch.float),
        edge_index=torch.stack([src, dst]),
    )
    d.num_nodes = n_nodes
    return d


def _stamp(block, n_nodes, W, P, sizes):
    """Attach the three cross-panel attributes AFTER collation."""
    block.gene_ids = torch.arange(W, dtype=torch.long)
    block.panel_masks = torch.zeros((P, W), dtype=torch.bool)
    block.panel_masks[:, : W // 2] = True
    block.panel_id = torch.cat([
        torch.full((n,), i % P, dtype=torch.long) for i, n in enumerate(sizes)
    ])
    return block


# --------------------------------------------------------------------------- #
# naming: none of the three may trip the "batch" substring rule
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key", ["gene_ids", "panel_masks", "panel_id"])
def test_names_do_not_trip_the_inc_substring_rule(key):
    """
    `Data.__inc__` returns a nonzero increment for keys containing 'batch',
    which is what shifted `adata_batch_ids`. Confirm these three names are
    inert under that rule, so they are never offset per section.
    """
    d = _section(4, 6)
    assert d.__inc__(key, torch.zeros(3)) == 0
    # control: the rule really does still fire on a 'batch' name
    assert d.__inc__("adata_batch_ids", torch.zeros(3)) != 0


# --------------------------------------------------------------------------- #
# the shape-collision hazard
# --------------------------------------------------------------------------- #

def test_gene_ids_is_not_a_node_attr_when_W_differs_from_N():
    """The ordinary case: W != N, so `gene_ids` must pass through untouched."""
    sizes = (7, 5)
    block = Batch.from_data_list([_section(n, 6, seed=i) for i, n in enumerate(sizes)])
    n_nodes, W, P = sum(sizes), 6, 2
    block = Data(**{k: v for k, v in block.to_dict().items() if k != "ptr"})
    _stamp(block, n_nodes, W, P, sizes)
    assert not block.is_node_attr("gene_ids")
    assert not block.is_node_attr("panel_masks")
    assert block.is_node_attr("panel_id")


def test_gene_ids_IS_misclassified_when_W_equals_N():
    """
    The landmine, asserted so the mitigation is justified by evidence rather
    than caution: with W == num_nodes, PyG's shape test alone makes a flat
    `gene_ids[W]` a node attribute. Nothing warns.
    """
    sizes = (7, 5)
    n_nodes = sum(sizes)
    block = Batch.from_data_list([_section(n, 6, seed=i) for i, n in enumerate(sizes)])
    block = Data(**{k: v for k, v in block.to_dict().items() if k != "ptr"})
    _stamp(block, n_nodes, W=n_nodes, P=2, sizes=sizes)      # W == N on purpose
    assert block.is_node_attr("gene_ids"), (
        "expected the shape collision; if this now fails, PyG changed its "
        "node-attr inference and the [1, W] mitigation may be unnecessary"
    )


@pytest.mark.parametrize("W", [12, 30])
def test_gene_ids_as_1xW_is_immune_to_the_collision(W):
    """
    THE MITIGATION, and why the real code stores `gene_ids` with a leading
    singleton dim. `[1, W]` is classified on `size(0) == num_nodes`, so it can
    only be misclassified when a block holds exactly ONE cell — which cannot
    happen. `[W]` collides whenever W == N, which at corpus scale (thousands of
    genes, thousands of cells per block) is an ordinary coincidence, not an
    edge case.

    `panel_masks[P, W]` needs no such treatment: it would require P == N, i.e.
    as many panels in the block as cells.
    """
    sizes = (7, 5)
    n_nodes = sum(sizes)
    block = Batch.from_data_list([_section(n, W, seed=i) for i, n in enumerate(sizes)])
    block = Data(**{k: v for k, v in block.to_dict().items() if k != "ptr"})
    ids = torch.arange(W, dtype=torch.long).unsqueeze(0)      # [1, W]
    block.gene_ids = ids.clone()

    assert not block.is_node_attr("gene_ids")
    if W == n_nodes:
        # the exact case that breaks the flat layout
        flat = Data(**{k: v for k, v in block.to_dict().items()})
        flat.gene_ids = torch.arange(W, dtype=torch.long)
        assert flat.is_node_attr("gene_ids")

    mb = next(iter(NeighborLoader(block, num_neighbors=[2], batch_size=4,
                                  shuffle=False)))
    torch.testing.assert_close(mb.gene_ids, ids)
    assert mb.gene_ids.shape == (1, W)


# --------------------------------------------------------------------------- #
# NeighborLoader round trip
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("W", [6, 12])
def test_round_trip_preserves_gene_ids_and_masks(W):
    """
    The contract that matters: after NeighborLoader, `gene_ids` and
    `panel_masks` must be IDENTICAL to what went in (not sliced, not
    concatenated per section), while `panel_id` must be sliced to the
    mini-batch's nodes so `panel_masks[panel_id]` lines up per cell.
    """
    sizes = (37, 23)
    n_nodes = sum(sizes)
    P = 2
    block = Batch.from_data_list([_section(n, W, seed=i) for i, n in enumerate(sizes)])
    block = Data(**{k: v for k, v in block.to_dict().items() if k != "ptr"})
    _stamp(block, n_nodes, W, P, sizes)
    gene_ids_in = block.gene_ids.clone()
    masks_in = block.panel_masks.clone()

    loader = NeighborLoader(block, num_neighbors=[2], batch_size=8, shuffle=False)
    mb = next(iter(loader))

    torch.testing.assert_close(mb.gene_ids, gene_ids_in)
    assert mb.panel_masks.shape == (P, W)
    torch.testing.assert_close(mb.panel_masks, masks_in)

    # panel_id is per-cell, so it MUST have been sliced down.
    assert mb.panel_id.numel() == mb.num_nodes < n_nodes
    assert mb.x.shape == (mb.num_nodes, W)

    # the whole point: a per-cell mask recovered by gather, aligned with x
    per_cell = mb.panel_masks[mb.panel_id]
    assert per_cell.shape == mb.x.shape
    assert per_cell.dtype == torch.bool


def test_gathered_mask_matches_the_originating_section():
    """
    End to end on the property union+mask depends on: each cell's recovered
    mask must be its own SECTION's panel mask. Uses distinguishable per-panel
    masks so a mix-up cannot pass.
    """
    sizes = (37, 23)
    W, P = 10, 2
    block = Batch.from_data_list([_section(n, W, seed=i) for i, n in enumerate(sizes)])
    block = Data(**{k: v for k, v in block.to_dict().items() if k != "ptr"})
    block.gene_ids = torch.arange(W, dtype=torch.long)
    # panel 0 measures the first 4 genes, panel 1 the last 6 — disjoint, so any
    # confusion between them is visible.
    block.panel_masks = torch.zeros((P, W), dtype=torch.bool)
    block.panel_masks[0, :4] = True
    block.panel_masks[1, 4:] = True
    block.panel_id = torch.cat([torch.full((n,), i, dtype=torch.long)
                                for i, n in enumerate(sizes)])
    # `batch` (per-node section index) is what PyG built during collation.
    section_of_node = block.batch.clone()

    loader = NeighborLoader(block, num_neighbors=[-1], batch_size=16, shuffle=False)
    for mb in loader:
        expected_panel = section_of_node[mb.n_id]
        torch.testing.assert_close(mb.panel_id, expected_panel)
        got = mb.panel_masks[mb.panel_id]
        assert got[expected_panel == 0][:, :4].all()
        assert not got[expected_panel == 0][:, 4:].any()
        assert got[expected_panel == 1][:, 4:].all()
        assert not got[expected_panel == 1][:, :4].any()
