"""
Tier 1 is four config changes and two pieces of new machinery.

The machinery is what needs testing, because both paths were unreachable
before: `configure_optimizers` returned a bare optimizer, so this model has
never handed Lightning a scheduler; and `ResidualVQ_Squint` hard-coded
`threshold_ema_dead_code if i == 0 else 0`, so dead-code revival above level 0
had no way to be switched on.

The config changes are tested as a DIFF against `corpus-holdout` rather than by
asserting absolute values. Tier 1's whole claim is that it changes four things
and nothing else -- an attributable delta at a fixed 200,000-step budget -- so
"nothing else moved" is the property worth pinning, and it is the one a future
edit to the shared reference stack would silently break.
"""
import importlib.util
import sys

import pytest
import torch

from vqniche.models.base_model import BaseModel
from vqniche.modules import get_valid_params, get_vq_class


def _rs():
    """Load the driver as a module without running its CLI."""
    sys.argv = ["run_squint.py"]
    spec = importlib.util.spec_from_file_location("rs", "examples/run_squint.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rs():
    return _rs()


# --------------------------------------------------------------------------
# The LR schedule
# --------------------------------------------------------------------------

class _Sched:
    """
    Borrows the real methods rather than re-deriving the curve, so the test
    fails if the schedule changes. Constructing a `BaseModel` needs an
    encoder and a decoder stack; the schedule needs four scalars.
    """
    _schedule_horizon = BaseModel._schedule_horizon
    _lr_lambda = BaseModel._lr_lambda

    def __init__(self, warmup=2_000, floor=0.05, total=200_000, trainer=None):
        self.warmup_steps = warmup
        self.min_lr_ratio = floor
        self.total_steps = total
        self.trainer = trainer
        self._horizon = None if total is None else total


def test_warmup_rises_from_nonzero_to_exactly_one():
    s = _Sched(warmup=2_000)
    # (step + 1) / warmup, so the first step is not a zero-LR no-op -- a
    # genuine 0 would waste the step and, with `fused=True`, has in the past
    # tripped optimizers that assume a non-zero LR.
    assert s._lr_lambda(0) == pytest.approx(1 / 2_000)
    assert s._lr_lambda(999) == pytest.approx(0.5)
    assert s._lr_lambda(1_999) == pytest.approx(1.0)
    # The boundary must not double-count: step 2000 is the first decay step
    # and cos(0) == 1, so it is still exactly the base LR.
    assert s._lr_lambda(2_000) == pytest.approx(1.0)


def test_cosine_decays_monotonically_to_the_floor_and_clamps():
    s = _Sched(warmup=2_000, floor=0.05, total=200_000)
    xs = [s._lr_lambda(k) for k in range(2_000, 200_001, 1_000)]
    assert all(b <= a for a, b in zip(xs, xs[1:], strict=False)), "decay is not monotone"
    assert xs[-1] == pytest.approx(0.05)
    # Past the horizon the multiplier holds at the floor rather than turning
    # back up, which an unclamped cosine would do.
    assert s._lr_lambda(400_000) == pytest.approx(0.05)


def test_floor_of_zero_reaches_zero():
    assert _Sched(warmup=10, floor=0.0, total=1_000)._lr_lambda(1_000) == pytest.approx(0.0)


def test_horizon_prefers_the_explicit_total_steps():
    # `trainer.estimated_stepping_batches` is unreliable on the streaming
    # backend, so an explicit value must win even when a trainer is present.
    class _T:
        estimated_stepping_batches = 12
    assert _Sched(total=200_000, trainer=_T())._schedule_horizon() == 200_000


@pytest.mark.parametrize("est", [None, float("inf"), 0, -5])
def test_horizon_refuses_a_useless_trainer_estimate(est):
    class _T:
        estimated_stepping_batches = est
    s = _Sched(total=None, trainer=_T())
    with pytest.raises(ValueError, match="finite step horizon"):
        s._schedule_horizon()


def test_horizon_falls_back_to_the_trainer_when_usable():
    class _T:
        estimated_stepping_batches = 5_000
    assert _Sched(total=None, trainer=_T())._schedule_horizon() == 5_000


# --------------------------------------------------------------------------
# Per-level dead-code revival
# --------------------------------------------------------------------------

_VQ = {
    "vq_name": "ResidualVQ_Squint", "num_quantizers": 2, "codebook_size": [30, 90],
    "use_cosine_sim": True, "ema_update": True, "decay": 0.8, "eps": 1e-05,
    "threshold_ema_dead_code": 2, "kmeans_init": True, "kmeans_iters": 10,
    "sync_kmeans": True, "commitment_weight": 0.0, "sample_codebook_temp": 0.0,
    "codebook_diversity_loss_weight": 0.0, "codebook_diversity_temperature": 100.0,
    "dim": 64,
}


def _build_vq(**extra):
    p = {**_VQ, **extra}
    cls = get_vq_class(p["vq_name"])
    valid = get_valid_params(cls, p)
    # `get_valid_params` filters to the constructor signature, so a flag
    # missing from the signature would be dropped SILENTLY rather than
    # raising -- assert it survived the filter before trusting the result.
    for k in extra:
        assert k in valid, f"{k} was filtered out; not in the signature?"
    return cls(**valid)


def test_level_1_revival_is_off_by_default():
    thresholds = [int(l._codebook.threshold_ema_dead_code) for l in _build_vq().layers]
    assert thresholds == [2, 0]


def test_dead_code_all_levels_arms_every_level():
    vq = _build_vq(dead_code_all_levels=True)
    assert [int(l._codebook.threshold_ema_dead_code) for l in vq.layers] == [2, 2]


def test_forward_contract_is_unchanged_by_the_flag():
    """
    The flag only touches a dead-code threshold, so every shape the encoder
    and the loss dispatcher depend on must be identical with and without it.
    Asserted as an equality between the two rather than against literals, so
    it stays true if the commit-loss shape itself is ever revised.
    """
    torch.manual_seed(0)
    z = torch.randn(256, 64)
    shapes = []
    for extra in ({}, {"dead_code_all_levels": True}):
        z_q, idx, commit = _build_vq(**extra)(z)
        assert z_q.shape == z.shape
        assert idx.shape == (256, 2)          # one assignment per level
        shapes.append((z_q.shape, idx.shape, commit.shape))
    assert shapes[0] == shapes[1]


# --------------------------------------------------------------------------
# `_patch_tier1` as a diff against `corpus-holdout`
# --------------------------------------------------------------------------

def test_tier1_changes_exactly_the_four_intended_things(rs):
    base = rs.VARIANTS["corpus-holdout"]["build"]()
    tier1 = rs.VARIANTS["corpus-holdout-tier1"]["build"]()

    # 1. loss rebalance -- adjacency was 55% of the objective at 1000.
    assert base["model"]["loss_params"]["loss_kwargs"]["wt_adj_reconstr"] == 1000.0
    assert tier1["model"]["loss_params"]["loss_kwargs"]["wt_adj_reconstr"] == 150.0

    # The commit weights are deliberately NOT touched: dropping adjacency
    # 6.7x already raises their relative influence 6.7x, and changing both
    # would make the effect unattributable.
    for k in ("wt_commit_cell", "wt_commit_niche"):
        assert tier1["model"]["loss_params"]["loss_kwargs"][k] == \
               base["model"]["loss_params"]["loss_kwargs"][k]

    # 2. symmetric codebook diversity -- was 10.0 cell / 0.0 niche.
    b_enc, t_enc = base["model"]["encoder_params"], tier1["model"]["encoder_params"]
    assert b_enc["vq_cell_params"]["codebook_diversity_loss_weight"] == 10.0
    assert b_enc["vq_niche_params"]["codebook_diversity_loss_weight"] == 0.0
    assert {t_enc[s]["codebook_diversity_loss_weight"]
            for s in ("vq_cell_params", "vq_niche_params")} == {0.0}

    # 3. dead-code revival on level 1, both branches.
    for s in ("vq_cell_params", "vq_niche_params"):
        assert "dead_code_all_levels" not in b_enc[s]
        assert t_enc[s]["dead_code_all_levels"] is True

    # 4. LR warmup + cosine decay, horizon taken from the step budget.
    assert "lr_schedule" not in base["model"]["optimizer_params"]
    opt = tier1["model"]["optimizer_params"]
    assert opt["lr_schedule"] == "cosine"
    assert opt["total_steps"] == tier1["trainer"]["max_steps"] == 200_000
    assert 0 < opt["warmup_steps"] < opt["total_steps"]

    # ...and the budget itself is untouched, which is what makes the run
    # comparable to the one being improved on.
    assert tier1["trainer"]["max_steps"] == base["trainer"]["max_steps"]
    assert tier1["model"]["optimizer_params"]["lr"] == base["model"]["optimizer_params"]["lr"]
    assert tier1["dataset"] == base["dataset"]
    assert tier1["datamodule"] == base["datamodule"]


def test_tier1_does_not_leak_into_the_baseline(rs):
    """
    `_patch_tier1` mutates in place, and the two variants share
    `_r0_reference_stack`. If the builders returned anything shared, running
    tier1 first would silently rewrite the baseline it is measured against.
    """
    rs.VARIANTS["corpus-holdout-tier1"]["build"]()
    base = rs.VARIANTS["corpus-holdout"]["build"]()
    assert base["model"]["loss_params"]["loss_kwargs"]["wt_adj_reconstr"] == 1000.0
    assert base["model"]["encoder_params"]["vq_cell_params"][
        "codebook_diversity_loss_weight"] == 10.0
    assert "dead_code_all_levels" not in base["model"]["encoder_params"]["vq_cell_params"]
    assert "lr_schedule" not in base["model"]["optimizer_params"]


def test_tier1_needs_a_step_budget(rs):
    """Without `max_steps` the cosine schedule has no horizon -- fail loudly."""
    cfg = rs._patch_dual_hst_corpus(rs._r0_reference_stack(), batch_size=512)
    cfg["trainer"].pop("max_steps", None)
    with pytest.raises(ValueError, match="step horizon"):
        rs._patch_tier1(cfg)


def test_tier1_refuses_dead_code_flag_on_a_single_codebook(rs):
    """
    The flag is per-LEVEL. On a plain `VectorQuantize` it would be filtered
    out by `get_valid_params` and vanish, so the patch has to refuse rather
    than appear to have applied.
    """
    cfg = rs._patch_dual_hst_corpus(rs._r0_reference_stack(), batch_size=512)
    cfg = rs._patch_step_budget(cfg, max_steps=1_000)
    cfg["model"]["encoder_params"]["vq_cell_params"]["vq_name"] = "VectorQuantize"
    with pytest.raises(ValueError, match="ResidualVQ_Squint"):
        rs._patch_tier1(cfg)
