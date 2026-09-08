"""
Tier 2a gives the ENCODER batch information. Two paths need testing because
neither could previously run.

Encoder FiLM has never actually worked on the streaming backend, and the way
it failed was SILENT. `encoder_conditions` carrying the batch one-hot is
assembled only in `initialize_databatch`, which streaming skips because it
would collate the corpus; the transform stamps `cell_batch_id` as a zero-width
placeholder. So `condition_dim` bound to 0, FiLM was built with a condition
carrying nothing, and the model trained exactly like the unconditioned one
while every log line said conditioning was on. The width now comes from the
batch count and the one-hot is derived per mini-batch from the block's
`adata_batch_ids`.

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


# --------------------------------------------------------------------------
# The batch-count helper, which is where the first Tier 2a smoke run failed
# --------------------------------------------------------------------------

def test_n_distinct_batches_prefers_the_explicit_streaming_count(rs):
    """
    The streaming probe is built from ONE section, so `adata_batch_ids.max()`
    on it would only ever see that section's own batch. It carries the
    corpus-wide count explicitly and that must win.
    """
    class _Probe:
        n_distinct_batches = 416
        adata_batch_ids = torch.tensor([7])      # would give 8 if consulted

    assert rs._n_distinct_batches(_Probe()) == 416


def test_n_distinct_batches_falls_back_to_ids_in_memory(rs):
    class _InMemory:
        adata_batch_ids = torch.tensor([0, 3, 2, 3])

    assert rs._n_distinct_batches(_InMemory()) == 4


def test_n_distinct_batches_rejects_an_object_carrying_neither(rs):
    """
    Passing the DATASET BLOB rather than the data batch is what broke the
    first Tier 2a smoke run: `OnDiskDatasetBlob` has neither attribute, and
    the bare AttributeError said nothing about which object was wrong.
    """
    class _Blob:
        pass

    with pytest.raises(TypeError, match="neither"):
        rs._n_distinct_batches(_Blob())


# ---------------------------------------------------------------------------
# Predict-time batch identity. Both defects below were SILENT and both made
# encoder conditioning inert, which is this module's subject.
# ---------------------------------------------------------------------------

class _StubBlob:
    """Minimal blob exposing only what the label-map path touches."""

    def __init__(self, labels, rels):
        self._labels, self._rels = labels, rels

    def __len__(self):
        return len(self._labels)

    def get_batch_labels(self):
        return list(self._labels)

    def section_rows_for(self, names):
        idx = {r: i for i, r in enumerate(self._rels)}
        return [idx[n] for n in names]

    def batch_label_to_dense(self, rows=None):
        labs = (sorted(set(self._labels)) if rows is None
                else sorted({self._labels[int(r)] for r in rows}))
        return {lbl: i for i, lbl in enumerate(labs)}


@pytest.fixture
def stub_blob():
    # 6 sections, composite labels as `batch_key='dataset_batch'` produces.
    labels = ["10_batch0", "10_batch1", "11_batch0",
              "11_batch1", "12_batch0", "12_batch1"]
    rels = [f"d{i}/s{i}.h5ad" for i in range(6)]
    return _StubBlob(labels, rels)


class _FakeCkpt:
    """A checkpoint carrying just the two widths the map is verified against."""

    def __init__(self, cov_dim, enc_dim):
        self.dims = {"decoder_covariate_dim": cov_dim,
                     "encoder_batch_condition_dim": enc_dim}

    def write(self, tmp_path):
        p = tmp_path / "fake.ckpt"
        torch.save({"hyper_parameters": self.dims}, p)
        return str(p)


def test_predict_label_map_uses_the_train_restricted_map(rs, stub_blob, tmp_path):
    """
    Under a whole-section split the model is sized to the TRAIN labels only.
    Predict used the blob-wide map, which is both wider and differently
    numbered, so a FiLM encoder could not run and the covariate read wrong rows.
    """
    split = {"val": ["d4/s4.h5ad"], "test": ["d5/s5.h5ad"]}
    mp, src = rs._predict_label_map(tmp_path, split, stub_blob, ckpt_path=None)
    assert len(mp) == 4 and "split_sections" in src
    assert "12_batch0" not in mp and "12_batch1" not in mp


def test_predict_label_map_rejects_the_split_predict_rewrites(rs, stub_blob, tmp_path):
    """
    Predict repurposes `datamodule.split_sections` into {"predict": [...]}.
    Reading that instead of the training split claims only the shard's own
    sections as held out, so the rebuild comes out nearly blob-wide. The width
    check must refuse it rather than proceed.
    """
    rewritten = {"predict": ["d0/s0.h5ad"]}
    with pytest.raises(ValueError, match="No candidate batch label->dense map"):
        rs._predict_label_map(
            tmp_path, rewritten, stub_blob,
            ckpt_path=_FakeCkpt(4, 4).write(tmp_path),
        )


def test_predict_label_map_accepts_only_the_width_the_model_was_built_with(
        rs, stub_blob, tmp_path):
    split = {"val": ["d4/s4.h5ad"], "test": ["d5/s5.h5ad"]}
    ck = _FakeCkpt(4, 4).write(tmp_path)
    mp, _ = rs._predict_label_map(tmp_path, split, stub_blob, ckpt_path=ck)
    assert len(mp) == 4

    # A model sized to all 6 labels was trained without a split, so the
    # blob-wide map is the correct one for it -- same code, different run.
    ck6 = _FakeCkpt(6, 0).write(tmp_path)
    mp6, src6 = rs._predict_label_map(tmp_path, split, stub_blob, ckpt_path=ck6)
    assert len(mp6) == 6 and "blob-wide" in src6


def test_bare_uns_batch_never_matches_a_composite_keyed_map(stub_blob):
    """
    The label-domain mismatch, stated directly.

    With `batch_key='dataset_batch'` the map is keyed by `<dataset_id>_batch<N>`
    (what the streaming loader stamps TRAINING ids from), while `obs_batch`
    holds the bare `uns['batch']`. Densifying from the bare value assigns every
    cell the unknown id AND flags it unseen, so the covariate collapses to the
    mean embedding and FiLM sees a constant one-hot -- indistinguishable from a
    shard of genuinely novel batches, hence silent.
    """
    from vqniche.initializers.initialize import build_batch_one_hot_from_obs

    label_map = stub_blob.batch_label_to_dense(rows=[0, 1, 2, 3])

    bare = [["batch0"] * 2, ["batch1"] * 3]
    ids, one_hot, unseen = build_batch_one_hot_from_obs(
        obs_batch=bare, label_to_dense=label_map, unknown_label_dense_id=0)
    assert bool(unseen.all()) and int(ids.max()) == 0

    composite = [["10_batch0"] * 2, ["10_batch1"] * 3]
    ids, one_hot, unseen = build_batch_one_hot_from_obs(
        obs_batch=composite, label_to_dense=label_map, unknown_label_dense_id=0)
    assert not bool(unseen.any())
    assert ids.tolist() == [0, 0, 1, 1, 1]
    assert one_hot.shape[1] == 4
