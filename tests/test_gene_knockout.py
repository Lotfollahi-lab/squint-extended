"""
In-silico gene knockout (SQUINT_GENE_KO): the input edit and the split-input
niche path that lets an edit land on a cell AS ITSELF or AS A NEIGHBOUR.

Every property here holds by construction, so a failure is a wiring bug, not
a finding:

  * with no edit, the split path reproduces the default forward exactly;
  * the hand-written mean-SAGE equals PyG's SAGEConv, including nodes with no
    in-edges and graphs with or without self-loops;
  * a neighbour-only knockout leaves the cell branch untouched for every cell
    (the cell branch never sees neighbours), and moves the niche branch only
    where an edited cell is someone's neighbour;
  * a cell-only knockout moves the niche branch only on the edited cells;
  * configurations the split path does not model raise instead of
    approximating.
"""

import types

import pytest
import torch

from vqniche.encoders.vqniche_dual_encoder import VQNiche_Dual_Encoder
from vqniche.models.vqniche_dual import VQNiche_Dual

G = 12  # genes
N = 10  # cells
KO = 3  # knocked-out gene column


def _encoder(num_layers=1, separate_niche_mlp=True):
    torch.manual_seed(0)
    vq = {
        "vq_name": "ResidualVQ_Squint",
        "num_quantizers": 2,
        "codebook_size": [4, 6],
        "use_cosine_sim": True,
        "ema_update": True,
        "decay": 0.8,
        "kmeans_init": False,
        "commitment_weight": 0.0,
    }
    mlp = {"hidden_channels": [8, 8], "dropout": 0.0, "act": "relu", "norm": None}
    return VQNiche_Dual_Encoder(
        in_channels=G,
        mlp_params=mlp,
        niche_mlp_params=dict(mlp) if separate_niche_mlp else None,
        gnn_name="SAGEConv",
        gnn_params={
            "hidden_channels": 8,
            "num_layers": num_layers,
            "act_first": True,
            "activation": "relu",
            "norm": None,
            "dropout": 0.0,
        },
        vq_cell_params=dict(vq),
        vq_niche_params=dict(vq),
    ).eval()


def _graph(self_loops=True):
    src = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 2, 0]
    dst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2]
    if self_loops:
        src += list(range(N))
        dst += list(range(N))
    return torch.tensor([src, dst])


def _x():
    torch.manual_seed(1)
    x = torch.poisson(torch.rand(N, G) * 3)
    x[:, KO] = 0.0
    x[[2, 5], KO] = 4.0  # only cells 2 and 5 express the gene
    return x


def _ko(x):
    x = x.clone()
    x[:, KO] = 0.0
    return x


@pytest.mark.parametrize("self_loops", [True, False])
def test_split_path_without_edit_reproduces_default_forward(self_loops):
    enc, x, ei = _encoder(), _x(), _graph(self_loops)
    with torch.no_grad():
        ref = enc(batch_x=x, batch_edge_index=ei)
        split = enc(batch_x=x, batch_edge_index=ei, batch_x_nbr=x.clone())
    for a, b in zip(ref, split, strict=True):
        if a.is_floating_point():
            torch.testing.assert_close(a, b)
        else:
            assert torch.equal(a, b)


def test_manual_sage_matches_pyg_with_isolated_node():
    """Node 9 gets no in-edges: PyG's mean over an empty set is 0."""
    enc, x = _encoder(), _x()
    ei = torch.tensor([[1, 0, 3, 2, 5], [0, 1, 2, 3, 4]])
    with torch.no_grad():
        h = enc.niche_mlp_module(x)
        ref = enc.gnn_module(h, ei)
        got = enc._split_niche_gnn(h, x, ei, None, None)
    torch.testing.assert_close(got, ref)


def test_neighbour_only_knockout_leaves_cell_branch_untouched():
    enc, x, ei = _encoder(), _x(), _graph()
    with torch.no_grad():
        base = enc(batch_x=x, batch_edge_index=ei)
        nbr = enc(batch_x=x, batch_edge_index=ei, batch_x_nbr=_ko(x))
    z_mlp, z_gnn, z_q_cell, _, idx_cell, _ = range(6)
    assert torch.equal(base[z_mlp], nbr[z_mlp])
    assert torch.equal(base[z_q_cell], nbr[z_q_cell])
    assert torch.equal(base[idx_cell], nbr[idx_cell])

    # niche output moves exactly where an edited cell (2 or 5) is a NEIGHBOUR
    src, dst = ei
    has_edited_nbr = torch.zeros(N, dtype=torch.bool)
    for s, d in zip(src.tolist(), dst.tolist(), strict=True):
        if s != d and s in (2, 5):
            has_edited_nbr[d] = True
    moved = (base[z_gnn] - nbr[z_gnn]).abs().amax(dim=1) > 0
    assert torch.equal(moved, has_edited_nbr)


def test_cell_only_knockout_moves_niche_only_on_edited_cells():
    enc, x, ei = _encoder(), _x(), _graph()
    with torch.no_grad():
        base = enc(batch_x=x, batch_edge_index=ei)
        cell = enc(batch_x=_ko(x), batch_edge_index=ei, batch_x_nbr=x)
    edited = torch.zeros(N, dtype=torch.bool)
    edited[[2, 5]] = True
    moved_cell = (base[0] - cell[0]).abs().amax(dim=1) > 0
    moved_niche = (base[1] - cell[1]).abs().amax(dim=1) > 0
    assert torch.equal(moved_cell, edited)
    assert torch.equal(moved_niche, edited)


def test_unsupported_encoder_raises():
    x, ei = _x(), _graph()
    for enc in (_encoder(num_layers=2), _encoder(separate_niche_mlp=False)):
        with pytest.raises(NotImplementedError):
            enc(batch_x=x, batch_edge_index=ei, batch_x_nbr=x.clone())


# --------------------------------------------------------------------------- #
# the input edit itself (VQNiche_Dual._gene_knockout_inputs)
# --------------------------------------------------------------------------- #


def _edit(x, scope, cols=(KO,), gene_ids=None):
    model = types.SimpleNamespace(gene_knockout={"cols": list(cols), "scope": scope})
    return VQNiche_Dual._gene_knockout_inputs(model, x, gene_ids)


def test_no_knockout_is_a_passthrough():
    x = _x()
    model = types.SimpleNamespace()
    bx, bn = VQNiche_Dual._gene_knockout_inputs(model, x, None)
    assert bx is x and bn is None


@pytest.mark.parametrize(
    "scope,self_edited,nbr_edited", [("both", True, None), ("cell", True, False), ("nbr", False, True)]
)
def test_scopes_route_the_edit(scope, self_edited, nbr_edited):
    x = _x()
    bx, bn = _edit(x, scope)
    assert bool((bx[:, KO] == 0).all()) == self_edited
    if nbr_edited is None:
        assert bn is None
    else:
        assert bool((bn[:, KO] == 0).all()) == nbr_edited
    assert x[2, KO] == 4.0  # the input is never mutated


def test_narrowed_block_maps_vocab_columns():
    """A block carrying vocabulary columns [1, 3, 7, ...] edits the right one."""
    x = _x()
    gene_ids = torch.tensor([1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    bx, _ = _edit(x, "both", cols=(7,), gene_ids=gene_ids)
    assert torch.all(bx[:, 2] == 0)
    assert torch.equal(bx[:, [0, 1, 3]], x[:, [0, 1, 3]])
    with pytest.raises(ValueError):
        _edit(x, "both", cols=(2,), gene_ids=gene_ids)  # not in this block
