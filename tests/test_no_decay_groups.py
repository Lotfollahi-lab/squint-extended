"""
Weight decay was erasing the batch embedding in every corpus run.

Adam applies decay every step whether or not a parameter received gradient,
and its adaptive normalisation makes the decay step ~`lr` regardless of
magnitude -- so a zero-gradient parameter reaches EXACTLY 0.0 within ~1,000
steps at lr=7e-4, wd=1e-3. A section appears in roughly one block out of ~114
over a 200,000-step run, so 85% of the per-batch embedding rows were zero:
52/416 (baseline), 69/416 (tier1), 67/416 (tier2a). The decoder covariate is
the paper's batch-correction lever and was structurally disabled.

The property worth testing is not that the optimizer builds -- it is that an
exempt parameter SURVIVES a run of zero gradient, which is precisely what
fails today. `test_decay_erases...` / `test_exempt_parameter_survives...` are
the pair: same setup, one exempt and one not.
"""
import importlib.util
import sys

import pytest
import torch

from vqniche.models.base_model import BaseModel


def _rs():
    sys.argv = ["run_squint.py"]
    spec = importlib.util.spec_from_file_location("rs", "examples/run_squint.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rs():
    return _rs()


class _Model(torch.nn.Module):
    """Two named parameters, one matching the exempt pattern."""

    def __init__(self):
        super().__init__()
        self.batch_embedding = torch.nn.Embedding(8, 4)
        self.trunk = torch.nn.Linear(4, 4)


class _Grouper(_Model):
    """Borrows the real grouping method rather than re-deriving it."""

    _param_groups = BaseModel._param_groups

    def __init__(self, patterns, weight_decay=1e-3):
        super().__init__()
        self.no_decay_patterns = list(patterns or [])
        self.weight_decay = weight_decay


# --------------------------------------------------------------------------
# The mechanism
# --------------------------------------------------------------------------

def test_decay_erases_an_ungradiented_parameter():
    """
    The bug, reproduced. Not a hypothetical: this is why 85% of the corpus
    runs' batch-embedding rows read exactly zero.
    """
    w = torch.nn.Parameter(torch.randn(16) * 0.02)
    opt = torch.optim.Adam([w], lr=7e-4, weight_decay=1e-3)
    for _ in range(1_000):
        opt.zero_grad()
        w.grad = torch.zeros_like(w)
        opt.step()
    assert float(w.norm()) == pytest.approx(0.0, abs=1e-9)


def test_exempt_parameter_survives_the_same_run():
    w = torch.nn.Parameter(torch.randn(16) * 0.02)
    before = float(w.norm())
    opt = torch.optim.Adam([{"params": [w], "weight_decay": 0.0}], lr=7e-4)
    for _ in range(1_000):
        opt.zero_grad()
        w.grad = torch.zeros_like(w)
        opt.step()
    assert float(w.norm()) == pytest.approx(before, rel=1e-6)


# --------------------------------------------------------------------------
# The grouping
# --------------------------------------------------------------------------

def test_no_patterns_returns_the_legacy_single_group():
    out = _Grouper(None)._param_groups()
    # Not a list of dicts: the legacy path must stay byte-identical.
    assert not isinstance(out, list)
    # embedding.weight + trunk.weight + trunk.bias
    assert len(list(out)) == 3


def test_split_puts_the_matching_parameter_in_the_zero_decay_group():
    groups = _Grouper(["batch_embedding"])._param_groups()
    assert isinstance(groups, list) and len(groups) == 2
    decay, no_decay = groups
    assert decay["weight_decay"] == 1e-3
    assert no_decay["weight_decay"] == 0.0
    assert len(no_decay["params"]) == 1          # the embedding weight only
    assert no_decay["params"][0].shape == (8, 4)
    # every other parameter still decays
    assert len(decay["params"]) == 2             # Linear weight + bias


def test_a_pattern_matching_nothing_raises():
    """
    A silent no-op here would leave the very parameters it names decaying to
    zero -- the exact bug this feature exists to fix.
    """
    with pytest.raises(ValueError, match="matched no parameter"):
        _Grouper(["does_not_exist"])._param_groups()


def test_a_pattern_that_matches_only_the_absent_module_still_works():
    """
    Tier 1 has no `conditioning_module`; the default pattern list names it
    anyway so one list serves both tiers. `batch_embedding` supplies the hit.
    """
    groups = _Grouper(["batch_embedding", "conditioning_module"])._param_groups()
    assert len(groups[1]["params"]) == 1


# --------------------------------------------------------------------------
# `_patch_nodecay` as a diff
# --------------------------------------------------------------------------

def test_nodecay_changes_exactly_one_thing(rs):
    t1 = rs.VARIANTS["corpus-holdout-tier1"]["build"]()
    nd = rs.VARIANTS["corpus-holdout-tier1-nodecay"]["build"]()

    assert "no_decay_patterns" not in t1["model"]["optimizer_params"]
    assert nd["model"]["optimizer_params"]["no_decay_patterns"] == [
        "batch_embedding", "conditioning_module"]

    # weight_decay itself is UNCHANGED -- the fix is about which parameters
    # it applies to, not how strong it is.
    assert nd["model"]["optimizer_params"]["weight_decay"] == \
        t1["model"]["optimizer_params"]["weight_decay"] == 0.001

    # and nothing else moved, or the comparison against Tier 1 is void
    assert nd["trainer"]["max_steps"] == t1["trainer"]["max_steps"] == 200_000
    assert nd["model"]["loss_params"] == t1["model"]["loss_params"]
    assert nd["model"]["encoder_params"] == t1["model"]["encoder_params"]
    assert nd["dataset"] == t1["dataset"]
    # crucially: still NO encoder FiLM. This variant tests whether Tier 1's
    # OWN mechanism works once it stops being erased.
    assert nd["model"]["encoder_params"].get("conditioning_params") is None


def test_nodecay_does_not_leak_into_tier1(rs):
    rs.VARIANTS["corpus-holdout-tier1-nodecay"]["build"]()
    t1 = rs.VARIANTS["corpus-holdout-tier1"]["build"]()
    assert "no_decay_patterns" not in t1["model"]["optimizer_params"]
