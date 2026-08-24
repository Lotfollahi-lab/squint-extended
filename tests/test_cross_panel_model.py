"""
Cross-panel model primitives: the vocabulary gather, the masked softmax and the
masked-sum NB.

These are the two changes in the whole cross-panel effort that fail SILENTLY
rather than crashing, which is why they get direct unit tests before being wired
into the model:

  * the softmax normalises across the gene axis, so unmasked it leaks
    probability mass onto genes a section never measured and deflates every
    measured prediction;
  * the NB summed over all columns, so unmeasured zeros would be scored as if
    they had been observed.

The other thing asserted throughout: with no mask and no gather, every path is
byte-identical to the single-panel behaviour. That is the defence for the
paper's tested code path, and it is exact here rather than approximate -- an
all-ones mask multiplies the NB term by 1, and a full-width gather is an
identity permutation.
"""
import pytest
import torch

from vqniche.decoders import MLPSoftmax
from vqniche.loss import nb_attribute_reconstruction_loss
from vqniche.modules import MLP, ConditionalMLP

V = 12          # gene vocabulary
W = 7           # genes carried by this block
H = 5           # hidden width
N = 4           # cells


@pytest.fixture
def ids():
    """A sorted, strictly increasing subset of the vocabulary, as blocks give."""
    return torch.tensor([0, 2, 3, 5, 8, 10, 11], dtype=torch.long)


def _pad_to_vocab(x_block, ids, vocab=V):
    """The dense reference: scatter a block back to full vocabulary width."""
    out = x_block.new_zeros((x_block.shape[0], vocab))
    out[:, ids] = x_block
    return out


# --------------------------------------------------------------------------- #
# input-side gather
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cls", [MLP, ConditionalMLP])
def test_input_gather_equals_the_dense_reference(cls, ids):
    """
    `F.linear(x_block, W[:, ids])` must equal padding the block up to V and
    using the full weight. If this drifts, every latent is wrong.
    """
    torch.manual_seed(0)
    mlp = cls(in_channels=V, hidden_channels=[H, 3])
    x_block = torch.randn(N, W)

    got = mlp(x_block, gene_ids=ids.unsqueeze(0))
    want = mlp(_pad_to_vocab(x_block, ids))
    torch.testing.assert_close(got, want)


def test_input_gather_accepts_both_id_layouts(ids):
    """Blocks stamp `gene_ids` as [1, W]; a bare [W] must work identically."""
    torch.manual_seed(0)
    mlp = MLP(in_channels=V, hidden_channels=[H, 3])
    x_block = torch.randn(N, W)
    torch.testing.assert_close(
        mlp(x_block, gene_ids=ids.unsqueeze(0)), mlp(x_block, gene_ids=ids)
    )


def test_gradients_reach_only_the_gathered_columns(ids):
    """
    A gene absent from a block must receive no update that step -- it was not
    observed. Anything else would drift unrelated genes on every batch.
    """
    torch.manual_seed(0)
    mlp = MLP(in_channels=V, hidden_channels=[H])
    mlp(torch.randn(N, W), gene_ids=ids).sum().backward()

    grad = mlp.lins[0].weight.grad
    assert grad is not None
    touched = grad.abs().sum(dim=0) > 0
    expected = torch.zeros(V, dtype=torch.bool)
    expected[ids] = True
    torch.testing.assert_close(touched, expected)


def test_no_gene_ids_is_unchanged():
    """The single-panel path: no gather, no behaviour change."""
    torch.manual_seed(0)
    mlp = MLP(in_channels=V, hidden_channels=[H, 3])
    x = torch.randn(N, V)
    torch.testing.assert_close(mlp(x), mlp(x, gene_ids=None))


def test_full_width_gather_is_the_identity(ids):
    """
    A block spanning the whole vocabulary must skip the gather entirely: a
    sorted index set of size V over [0, V) can only be arange(V).
    """
    torch.manual_seed(0)
    mlp = MLP(in_channels=V, hidden_channels=[H])
    x = torch.randn(N, V)
    all_ids = torch.arange(V)
    torch.testing.assert_close(mlp(x, gene_ids=all_ids), mlp(x))


# --------------------------------------------------------------------------- #
# output-side gather + masked softmax
# --------------------------------------------------------------------------- #

def _decoder(out_channels=V, hidden=(3,)):
    return MLPSoftmax(
        in_channels=H, out_channels=out_channels,
        mlp_params={"hidden_channels": list(hidden), "dropout": 0.0,
                    "act": "gelu", "norm": None},
    )


def test_output_gather_selects_the_right_genes(ids):
    """
    Gathering the output layer's ROWS must equal running the full V-wide layer
    and then selecting those columns -- done pre-softmax, so compare logits via
    a decoder with the mask off and temperature 1.
    """
    torch.manual_seed(0)
    dec = _decoder()
    z = torch.randn(N, H)
    depth = torch.ones(N)

    full = dec(x=z, read_depth=depth)                       # softmax over V
    narrow = dec(x=z, read_depth=depth, gene_ids=ids)       # softmax over W
    # The softmax normalisers differ (V vs W genes), so compare RATIOS between
    # two genes, which are invariant to the normaliser.
    r_full = full[:, ids[0]] / full[:, ids[1]]
    r_narrow = narrow[:, 0] / narrow[:, 1]
    torch.testing.assert_close(r_full, r_narrow)
    assert narrow.shape == (N, len(ids))


def test_masked_softmax_puts_zero_mass_on_unmeasured_genes(ids):
    torch.manual_seed(0)
    dec = _decoder()
    z = torch.randn(N, H)
    depth = torch.full((N,), 7.0)
    mask = torch.zeros(N, len(ids), dtype=torch.bool)
    mask[:, :4] = True                                       # 4 of 7 measured

    out = dec(x=z, read_depth=depth, gene_ids=ids, gene_mask=mask)
    assert torch.all(out[~mask] == 0.0)
    # and the measured mass is exactly the read depth
    torch.testing.assert_close(out.sum(dim=-1), depth)


def test_unmasked_softmax_leaks_mass_which_is_the_bug(ids):
    """
    The failure this masking exists to prevent, asserted so the fix is
    justified by evidence: without a mask the unmeasured columns absorb
    probability, so the measured genes' predictions come out DEFLATED.
    """
    torch.manual_seed(0)
    dec = _decoder()
    z = torch.randn(N, H)
    depth = torch.full((N,), 7.0)
    mask = torch.zeros(N, len(ids), dtype=torch.bool)
    mask[:, :4] = True

    leaky = dec(x=z, read_depth=depth, gene_ids=ids)
    fixed = dec(x=z, read_depth=depth, gene_ids=ids, gene_mask=mask)
    assert torch.all(leaky[mask] < fixed[mask])
    assert float(leaky[~mask].sum()) > 0.0        # mass really did leak


def test_mask_shape_mismatch_raises(ids):
    dec = _decoder()
    with pytest.raises(ValueError, match="must match the decoder output"):
        dec(x=torch.randn(N, H), read_depth=torch.ones(N), gene_ids=ids,
            gene_mask=torch.zeros(N, len(ids) - 1, dtype=torch.bool))


def test_all_unmeasured_row_raises_instead_of_returning_nan(ids):
    """softmax over an all -inf row is NaN; refuse rather than propagate it."""
    dec = _decoder()
    mask = torch.ones(N, len(ids), dtype=torch.bool)
    mask[1] = False
    with pytest.raises(ValueError, match="no measured gene"):
        dec(x=torch.randn(N, H), read_depth=torch.ones(N), gene_ids=ids,
            gene_mask=mask)


def test_decoder_without_cross_panel_args_is_unchanged():
    torch.manual_seed(0)
    dec = _decoder()
    z, depth = torch.randn(N, H), torch.rand(N) + 1
    torch.testing.assert_close(
        dec(x=z, read_depth=depth),
        dec(x=z, read_depth=depth, gene_ids=None, gene_mask=None),
    )


# --------------------------------------------------------------------------- #
# masked-sum NB
# --------------------------------------------------------------------------- #

def _nb_args(n_genes, seed=0):
    torch.manual_seed(seed)
    return {
        "pred_attr": torch.rand(N, n_genes) + 0.5,
        "target_attr": torch.randint(0, 5, (N, n_genes)).float(),
        "edge_index": torch.stack([torch.arange(N), torch.arange(N)]),
        "batch_size": N,
        "dispersion": torch.rand(n_genes) + 0.5,
    }


def test_all_ones_mask_is_bit_identical_to_unmasked():
    """
    The reduction change must be a no-op on the tested path. Bit-identical, not
    merely close: multiplying by a 1.0 mask cannot perturb the sum.
    """
    a = _nb_args(W)
    unmasked = nb_attribute_reconstruction_loss(**a)
    masked = nb_attribute_reconstruction_loss(
        **a, gene_mask=torch.ones(N, W, dtype=torch.bool))
    assert unmasked.item() == masked.item()


def test_masked_nb_ignores_unmeasured_columns():
    """
    Inject nonsense into the masked positions; the loss must not move. This is
    the property that makes zero-filled unmeasured genes safe.
    """
    a = _nb_args(W)
    mask = torch.zeros(N, W, dtype=torch.bool)
    mask[:, :4] = True
    before = nb_attribute_reconstruction_loss(**a, gene_mask=mask)

    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in a.items()}
    b["target_attr"][~mask] = 999.0
    b["pred_attr"][~mask] = 0.001
    after = nb_attribute_reconstruction_loss(**b, gene_mask=mask)
    torch.testing.assert_close(before, after)


def test_masked_nb_is_strictly_smaller_than_scoring_everything():
    """Dropping terms from a negative log-likelihood must reduce it."""
    a = _nb_args(W)
    mask = torch.zeros(N, W, dtype=torch.bool)
    mask[:, :4] = True
    assert (nb_attribute_reconstruction_loss(**a, gene_mask=mask).item()
            < nb_attribute_reconstruction_loss(**a).item())


def test_masked_nb_is_a_sum_not_a_mean():
    """
    Guards the decision explicitly: a mean over measured genes would divide the
    term by ~W and let the adjacency BCE (weight 1000) dominate reconstruction
    by ~3,000x at R0's magnitudes. Doubling the measured width must roughly
    double the loss, not leave it unchanged.
    """
    a = _nb_args(8)
    narrow = torch.zeros(N, 8, dtype=torch.bool)
    narrow[:, :2] = True
    wide = torch.zeros(N, 8, dtype=torch.bool)
    wide[:, :4] = True
    l_narrow = nb_attribute_reconstruction_loss(**a, gene_mask=narrow).item()
    l_wide = nb_attribute_reconstruction_loss(**a, gene_mask=wide).item()
    assert l_wide > 1.5 * l_narrow, (l_narrow, l_wide)


def test_nb_mask_shape_mismatch_raises():
    a = _nb_args(W)
    with pytest.raises(ValueError, match="must match the NB term"):
        nb_attribute_reconstruction_loss(
            **a, gene_mask=torch.ones(N, W - 1, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# the two model-side helpers
# --------------------------------------------------------------------------- #

def test_gather_genes_narrows_per_gene_parameters(ids):
    """`dispersion` / `dispersion_niche` are V-wide and must follow the block."""
    from vqniche.models.vqniche_dual import _gather_genes

    disp = torch.arange(V, dtype=torch.float)
    got = _gather_genes(disp, ids.unsqueeze(0))
    torch.testing.assert_close(got, ids.float())
    assert got.numel() == len(ids)


def test_gather_genes_is_a_noop_without_ids_or_at_full_width():
    from vqniche.models.vqniche_dual import _gather_genes

    disp = torch.arange(V, dtype=torch.float)
    assert _gather_genes(disp, None) is disp
    # full-width: a sorted size-V index set over [0, V) can only be arange(V)
    assert _gather_genes(disp, torch.arange(V)) is disp


def test_per_cell_gene_mask_gathers_from_the_panel_table():
    from torch_geometric.data import Data

    from vqniche.models.vqniche_dual import VQNiche_Dual

    blk = Data()
    blk.panel_masks = torch.tensor([[True, True, False],
                                    [False, True, True]])
    blk.panel_id = torch.tensor([0, 0, 1])
    got = VQNiche_Dual._per_cell_gene_mask(blk)
    assert got.shape == (3, 3)
    torch.testing.assert_close(got, blk.panel_masks[blk.panel_id])


def test_per_cell_gene_mask_is_none_on_the_single_panel_path():
    from torch_geometric.data import Data

    from vqniche.models.vqniche_dual import VQNiche_Dual

    assert VQNiche_Dual._per_cell_gene_mask(Data(x=torch.zeros(2, 3))) is None


def test_per_cell_gene_mask_refuses_half_a_panel_table():
    """
    Half the pair means block assembly changed without this consumer following.
    Training on silently no mask is the exact failure the mask prevents, so
    raise rather than degrade.
    """
    from torch_geometric.data import Data

    from vqniche.models.vqniche_dual import VQNiche_Dual

    blk = Data()
    blk.panel_masks = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="half the panel table"):
        VQNiche_Dual._per_cell_gene_mask(blk)
