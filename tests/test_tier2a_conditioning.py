"""
Tier 2a gives the ENCODER batch information. Two paths need testing because
neither could previously run.

Encoder FiLM has never executed on the streaming backend. `encoder_conditions`
carrying the batch one-hot is assembled only in `initialize_databatch`, which
streaming skips because it would collate the corpus -- so `conditioning_params`
was not merely off, it would have raised NameError on the `data_batch` that
the condition-dim binding referenced. The one-hot is now derived per mini-batch
from the block's `adata_batch_ids`.

The failure this guards against is silence: FiLM handed a zero-width or absent
condition trains exactly like the unconditioned model while every log line
still says conditioning is on.
"""
import importlib.util
import sys

import pytest
import torch

from vqniche.models.vqniche_dual import VQNiche_Dual
from vqniche.modules.film import FiLM


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
# The per-mini-batch batch one-hot
# --------------------------------------------------------------------------

class _Cond:
    """
    Borrows the real method rather than re-deriving it, so the test fails if
    the derivation changes. Building a `VQNiche_Dual` needs a full encoder and
    two decoders; the one-hot needs two attributes.
    """
    _encoder_batch_conditions = VQNiche_Dual._encoder_batch_conditions

    def __init__(self, dim):
        self.encoder_batch_condition_dim = dim


class _Batch:
    def __init__(self, ids, dtype=torch.float32):
        if ids is not None:
            self.adata_batch_ids = torch.as_tensor(ids)
        self.x = torch.zeros(1, 1, dtype=dtype)


def test_off_by_default_returns_none():
    # Every variant predating Tier 2a leaves the width at 0, and must be
    # byte-for-byte the unconditioned path.
    assert _Cond(0)._encoder_batch_conditions(_Batch([0, 1, 2])) is None


def test_one_hot_shape_and_content():
    out = _Cond(4)._encoder_batch_conditions(_Batch([0, 3, 1]))
    assert out.shape == (3, 4)
    assert out.dtype == torch.float32
    assert torch.equal(out.argmax(dim=1), torch.tensor([0, 3, 1]))
    # Exactly one hot per row -- a wrong num_classes would still produce a
    # plausible-looking matrix.
    assert torch.equal(out.sum(dim=1), torch.ones(3))


def test_dtype_follows_x():
    # FiLM concatenates this with encoder activations; a dtype mismatch under
    # bf16-mixed would raise deep inside the generator MLP.
    out = _Cond(3)._encoder_batch_conditions(_Batch([0, 1], dtype=torch.float64))
    assert out.dtype == torch.float64


def test_missing_batch_ids_raises_rather_than_silently_disabling():
    with pytest.raises(ValueError, match="no `adata_batch_ids`"):
        _Cond(4)._encoder_batch_conditions(_Batch(None))


def test_id_beyond_the_conditioning_width_raises():
    # The densification map and the condition dim disagreeing would one-hot
    # into the wrong column, or silently wrap.
    with pytest.raises(ValueError, match="exceeds the conditioning width"):
        _Cond(3)._encoder_batch_conditions(_Batch([0, 5]))


# --------------------------------------------------------------------------
# FiLM at identity init must be a no-op
# --------------------------------------------------------------------------

def test_identity_init_leaves_activations_unchanged():
    """
    `init_mode="identity"` is what makes Tier 2a attributable: at step 0 the
    model must be numerically the Tier 1 model, so any measured change comes
    from the conditioning learning something and not from a different
    initialisation.
    """
    torch.manual_seed(0)
    film = FiLM(in_channels=16, condition_dim=8, init_mode="identity",
                use_bias=True, use_residual=False, residual_weight=0.2)
    film.eval()
    h = torch.randn(32, 16)
    cond = torch.nn.functional.one_hot(torch.arange(32) % 8, 8).float()
    with torch.no_grad():
        assert torch.allclose(film(h, cond), h, atol=1e-6)


# --------------------------------------------------------------------------
# `_patch_tier2a` as a diff against Tier 1
# --------------------------------------------------------------------------

def test_tier2a_changes_exactly_the_two_intended_things(rs):
    t1 = rs.VARIANTS["corpus-holdout-tier1"]["build"]()
    t2 = rs.VARIANTS["corpus-holdout-tier2a"]["build"]()

    # 1. encoder conditioning, which did not exist before
    assert t1["model"]["encoder_params"].get("conditioning_params") is None
    cp = t2["model"]["encoder_params"]["conditioning_params"]
    assert cp["condition_list"] == ["cell_batch_id"]
    assert cp["init_mode"] == "identity"
    # condition_dim is bound in train() from the batch count, not at build
    # time -- asserting its absence pins that contract.
    assert "condition_dim" not in cp

    # 2. decoder covariate width
    assert t1["model"]["decoder_covariate_embed_dim"] == 16
    assert t2["model"]["decoder_covariate_embed_dim"] == 64

    # ...and nothing else. Tier 2a is measured against Tier 1, so a change to
    # the budget, LR or losses would make the comparison meaningless.
    assert t2["trainer"]["max_steps"] == t1["trainer"]["max_steps"] == 200_000
    assert t2["model"]["optimizer_params"] == t1["model"]["optimizer_params"]
    assert t2["model"]["loss_params"] == t1["model"]["loss_params"]
    assert t2["dataset"] == t1["dataset"]
    assert t2["datamodule"] == t1["datamodule"]
    for slot in ("vq_cell_params", "vq_niche_params"):
        assert (t2["model"]["encoder_params"][slot]
                == t1["model"]["encoder_params"][slot])
    # No adversary: that is Tier 2b, and a 416-way head fed ~3 classes per
    # block needs design work first.
    assert t2["model"].get("adversarial_alpha", 0.0) == 0.0
    assert not t2["model"].get("adversarial_batch_dim_request", False)


def test_tier2a_does_not_leak_into_tier1(rs):
    """Both builders share `_r0_reference_stack`; the patch mutates in place."""
    rs.VARIANTS["corpus-holdout-tier2a"]["build"]()
    t1 = rs.VARIANTS["corpus-holdout-tier1"]["build"]()
    assert t1["model"]["encoder_params"].get("conditioning_params") is None
    assert t1["model"]["decoder_covariate_embed_dim"] == 16


def test_tier2a_refuses_without_a_decoder_covariate(rs):
    cfg = rs._patch_step_budget(
        rs._patch_dual_hst_corpus(rs._BD(), batch_size=512), max_steps=1_000)
    cfg["model"].pop("decoder_covariate_dim_request", None)
    with pytest.raises(ValueError, match="no decoder covariate"):
        rs._patch_tier2a(cfg)


def test_tier2a_refuses_a_single_vq_config(rs):
    cfg = rs._patch_step_budget(rs._B(), max_steps=1_000)
    cfg["model"]["decoder_covariate_dim_request"] = True
    with pytest.raises(KeyError, match="dual-VQ"):
        rs._patch_tier2a(cfg)
