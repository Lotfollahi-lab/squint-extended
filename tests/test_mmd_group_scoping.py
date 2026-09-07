"""
The MMD batch term must align sections WITHIN a tissue, not across tissues.

Random section->block assignment puts more than one of the 18 training
tissues in 100% of blocks, and tissue is the strongest term in the
section-similarity regression (+0.223, against +0.077 for panel width and
+0.012 for assay). An unscoped MMD would therefore align cells across ORGANS
-- buying a batch-mixing metric by deleting the dominant biological signal.

The failure mode is silent in both directions: an unscoped loss still trains
and still reports a number, and a scoped loss whose group lookup is wrong
scopes to the wrong partition while looking identical. So the tests pin the
BEHAVIOUR -- that cross-group pairs contribute nothing and within-group pairs
do -- rather than the plumbing.
"""
import importlib.util
import sys

import pytest
import torch

from vqniche.loss.mmd_batch import mmd_batch_loss


def _rs():
    sys.argv = ["run_squint.py"]
    spec = importlib.util.spec_from_file_location("rs", "examples/run_squint.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rs():
    return _rs()


def _two_separated_batches(n=64, d=8, shift=6.0, seed=0):
    """Two batches whose distributions are far apart, so MMD is clearly > 0."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, d, generator=g)
    b = torch.randn(n, d, generator=g) + shift
    return torch.cat([a, b]), torch.cat([torch.zeros(n), torch.ones(n)]).long()


def test_unscoped_penalises_two_separated_batches():
    x, lab = _two_separated_batches()
    out = mmd_batch_loss(x, lab, wt_mmd_batch=1.0)
    assert float(out) > 0.1, "MMD should be large for well-separated batches"


def test_same_group_still_penalised():
    """Both batches in one tissue: this IS a batch effect, so it must count."""
    x, lab = _two_separated_batches()
    grp = torch.zeros_like(lab)                     # same tissue
    scoped = mmd_batch_loss(x, lab, wt_mmd_batch=1.0, mmd_group_labels=grp)
    unscoped = mmd_batch_loss(x, lab, wt_mmd_batch=1.0)
    assert float(scoped) == pytest.approx(float(unscoped), rel=1e-5)


def test_different_groups_contribute_nothing():
    """
    Two batches in DIFFERENT tissues. Their distributions differ because the
    organs differ, and forcing them together is exactly the damage the scope
    exists to prevent -- so the loss must be zero, not merely smaller.
    """
    x, lab = _two_separated_batches()
    grp = lab.clone()                               # batch 0 -> tissue 0, batch 1 -> tissue 1
    out = mmd_batch_loss(x, lab, wt_mmd_batch=1.0, mmd_group_labels=grp)
    assert float(out) == pytest.approx(0.0, abs=1e-9)


def test_scope_keeps_within_group_pairs_and_drops_cross_group():
    """
    Four batches, two tissues, two batches each. Only the two within-tissue
    pairs may count -- 2 of the 6 possible pairs.
    """
    g = torch.Generator().manual_seed(1)
    parts, labs, grps = [], [], []
    for b, (grp, shift) in enumerate([(0, 0.0), (0, 5.0), (1, 40.0), (1, 45.0)]):
        parts.append(torch.randn(48, 8, generator=g) + shift)
        labs.append(torch.full((48,), b))
        grps.append(torch.full((48,), grp))
    x = torch.cat(parts)
    lab = torch.cat(labs).long()
    grp = torch.cat(grps).long()
    scoped = float(mmd_batch_loss(x, lab, wt_mmd_batch=1.0, mmd_group_labels=grp))
    unscoped = float(mmd_batch_loss(x, lab, wt_mmd_batch=1.0))
    assert scoped > 0.0, "within-tissue pairs must still contribute"
    # The cross-tissue pairs are the far-apart ones, so dropping them must
    # LOWER the mean -- an equal value would mean the scope did nothing.
    assert scoped < unscoped


def test_scope_is_differentiable():
    x, lab = _two_separated_batches()
    x = x.clone().requires_grad_(True)
    mmd_batch_loss(x, lab, wt_mmd_batch=1.0,
                   mmd_group_labels=torch.zeros_like(lab)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0


def test_a_batch_spanning_two_groups_raises():
    """
    A batch is one section and a section has one tissue, so this cannot arise
    from real data -- it means the lookup is wrong, and scoping to the wrong
    partition would be invisible in the loss value.
    """
    x, lab = _two_separated_batches()
    grp = torch.zeros_like(lab)
    grp[:10] = 1                                    # batch 0 now spans 2 groups
    with pytest.raises(ValueError, match="spans multiple group labels"):
        mmd_batch_loss(x, lab, wt_mmd_batch=1.0, mmd_group_labels=grp)


# --------------------------------------------------------------------------
# The model-side lookup and the variant diff
# --------------------------------------------------------------------------

def test_group_lookup_gathers_per_cell(rs):
    from vqniche.models.vqniche_dual import VQNiche_Dual

    class _M:
        _mmd_group_labels = VQNiche_Dual._mmd_group_labels
        mmd_group_map = torch.tensor([5, 5, 7, 9])
    out = _M()._mmd_group_labels(torch.tensor([0, 2, 3, 1]))
    assert out.tolist() == [5, 7, 9, 5]


def test_group_lookup_refuses_an_out_of_range_id(rs):
    from vqniche.models.vqniche_dual import VQNiche_Dual

    class _M:
        _mmd_group_labels = VQNiche_Dual._mmd_group_labels
        mmd_group_map = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="outside mmd_group_map"):
        _M()._mmd_group_labels(torch.tensor([0, 5]))


def test_no_map_leaves_the_loss_globally_scoped(rs):
    from vqniche.models.vqniche_dual import VQNiche_Dual

    class _M:
        _mmd_group_labels = VQNiche_Dual._mmd_group_labels
        mmd_group_map = None
    assert _M()._mmd_group_labels(torch.tensor([0, 1])) is None


def test_tier2b_diff_against_ep3(rs):
    ep3 = rs.VARIANTS["corpus-holdout-nodecay-3ep"]["build"]()
    mmd = rs.VARIANTS["corpus-holdout-ep3-mmd"]["build"]()

    assert "mmd_batch_loss" not in ep3["model"]["loss_params"]["loss_names"]
    assert "mmd_batch_loss" in mmd["model"]["loss_params"]["loss_names"]
    assert mmd["model"]["loss_params"]["loss_kwargs"]["wt_mmd_batch"] == 500.0

    # 416 batches / 18 tissues, matching decoder_covariate_dim for this split
    gm = mmd["model"]["mmd_group_map"]
    assert len(gm) == 416 and len(set(gm)) == 18
    assert ep3["model"].get("mmd_group_map") is None

    # blocks widened, because at 3.9 sections only 56.4% of blocks contain a
    # same-tissue pair and the loss would be a no-op on the rest
    assert ep3["datamodule"]["max_cells_per_block"] == 900_000
    assert mmd["datamodule"]["max_cells_per_block"] == 1_900_000

    # budget, LR and the no-decay fix all carried through unchanged
    assert mmd["trainer"]["max_steps"] == ep3["trainer"]["max_steps"] == 562_687
    assert mmd["model"]["optimizer_params"] == ep3["model"]["optimizer_params"]
    assert mmd["model"]["encoder_params"] == ep3["model"]["encoder_params"]


def test_tier2b_does_not_leak_into_ep3(rs):
    rs.VARIANTS["corpus-holdout-ep3-mmd"]["build"]()
    ep3 = rs.VARIANTS["corpus-holdout-nodecay-3ep"]["build"]()
    assert "mmd_batch_loss" not in ep3["model"]["loss_params"]["loss_names"]
    assert ep3["model"].get("mmd_group_map") is None
    assert ep3["datamodule"]["max_cells_per_block"] == 900_000


def test_wt_override_refuses_a_variant_without_the_loss(rs):
    """
    A weight override on a variant with no MMD term would set a kwarg nothing
    reads -- a sweep that varies nothing, and looks like it worked.
    """
    import inspect
    src = inspect.getsource(rs.train)
    assert "--wt-mmd-batch given but" in src, "the guard is missing from train()"


def test_registry_requests_the_group_key(rs):
    """
    The dispatcher extracts exactly the keys the registry names. Omitting
    `mmd_group_labels` does not disable the scope -- it runs the loss
    GLOBALLY, silently, which at corpus scale aligns cells across organs.
    That is what happened on the first Tier 2b sweep.
    """
    import inspect

    from vqniche.models.base_model import BaseModel
    src = inspect.getsource(BaseModel.set_loss_fn_tuples)
    blk = src[src.index("loss_fn_name == 'mmd_batch_loss'"):]
    blk = blk[:blk.index("elif loss_fn_name ==")]
    assert "'mmd_group_labels'" in blk, "registry must request the scope key"


def test_loss_accepts_an_explicit_none_scope():
    """`_step` always sets the key, using None when no map is configured."""
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(32, 8, generator=g),
                   torch.randn(32, 8, generator=g) + 5])
    lab = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    out = mmd_batch_loss(x, lab, wt_mmd_batch=1.0, mmd_group_labels=None)
    assert isinstance(out, torch.Tensor)


def test_every_return_path_is_a_tensor():
    """
    A loss that falls off the end returns None, and the dispatcher's
    `torch.add` then raises without naming it. Pin the edge cases the corpus
    will actually hit: a single-batch mini-batch, and a block where every
    pair is cross-tissue so nothing is in scope.
    """
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(32, 8, generator=g),
                   torch.randn(32, 8, generator=g) + 5])
    lab = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    cases = {
        "two batches, unscoped": (x, lab, None),
        "one batch only": (x[:32], lab[:32], None),
        "all pairs out of scope": (x, lab, lab.clone()),
        "empty input": (x[:0], lab[:0], None),
    }
    for name, (xx, ll, gg) in cases.items():
        out = mmd_batch_loss(xx, ll, wt_mmd_batch=500.0, mmd_group_labels=gg)
        assert isinstance(out, torch.Tensor), f"{name} returned {type(out)}"


def test_dispatcher_names_a_loss_that_returns_non_tensor(rs):
    import inspect

    from vqniche.models.base_model import BaseModel
    src = inspect.getsource(BaseModel.criterion)
    assert "not a Tensor" in src and "loss_fn_name!r" in src, (
        "criterion must name the offending loss; an anonymous "
        "'must be Tensor, not NoneType' cost a debugging cycle"
    )
